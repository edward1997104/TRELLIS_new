import argparse
import math
import multiprocessing as mp
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import trimesh
from scipy.spatial import cKDTree
from tqdm import tqdm

_GLOBAL_POINTS = None
_GLOBAL_SQUARED = False
_GLOBAL_KDTREE_WORKERS = 1


def gather_files(input_dirs: List[str], suffix: str, recursive: bool = True) -> List[Path]:
    files: List[Path] = []
    for d in input_dirs:
        base = Path(d).expanduser().resolve()
        if not base.exists():
            raise FileNotFoundError(f"Input directory not found: {base}")
        if recursive:
            files.extend(sorted(base.rglob(f"*{suffix}")))
        else:
            files.extend(sorted(base.glob(f"*{suffix}")))
    files = [f for f in files if f.is_file()]
    if not files:
        raise ValueError(f"No {suffix} files found in the input directories.")
    return files


def scene_to_mesh(scene_or_mesh: trimesh.Scene | trimesh.Trimesh) -> trimesh.Trimesh:
    if isinstance(scene_or_mesh, trimesh.Trimesh):
        if scene_or_mesh.faces is None or len(scene_or_mesh.faces) == 0:
            raise ValueError("Loaded mesh has no faces.")
        return scene_or_mesh

    if not isinstance(scene_or_mesh, trimesh.Scene):
        raise ValueError(f"Unsupported loaded object type: {type(scene_or_mesh)}")

    meshes = []
    for g in scene_or_mesh.geometry.values():
        if isinstance(g, trimesh.Trimesh) and g.faces is not None and len(g.faces) > 0:
            meshes.append(g)
    if not meshes:
        raise ValueError("No triangular mesh geometry found in scene.")
    return trimesh.util.concatenate(meshes)


def normalize_points(points: np.ndarray) -> np.ndarray:
    # User-requested normalization:
    # 1) subtract point mean
    # 2) divide by longest axis length of the resulting bounding box
    points = points.astype(np.float32)
    points = points - points.mean(axis=0, keepdims=True)
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    longest_axis = float((bbox_max - bbox_min).max())
    if longest_axis <= 1e-12:
        raise ValueError("Degenerate point set: longest axis is ~0.")
    return points / longest_axis


def load_and_sample_points(obj_path: Path, num_points: int) -> np.ndarray:
    loaded = trimesh.load(str(obj_path), force="scene", process=False)
    mesh = scene_to_mesh(loaded)
    points = mesh.sample(num_points)
    return normalize_points(points)


def adjust_num_points(points: np.ndarray, num_points: int, rng: np.random.Generator) -> np.ndarray:
    n = points.shape[0]
    if n == num_points:
        return points
    if n > num_points:
        idx = rng.choice(n, size=num_points, replace=False)
    else:
        idx = rng.choice(n, size=num_points, replace=True)
    return points[idx]


def load_points_from_npz(
    npz_path: Path,
    points_key: str,
    num_points: int | None,
    renormalize: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    with np.load(str(npz_path), allow_pickle=False) as data:
        if points_key in data:
            points = data[points_key]
        else:
            fallback = None
            for k in data.files:
                arr = data[k]
                if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] == 3:
                    fallback = arr
                    break
            if fallback is None:
                raise KeyError(
                    f"Key '{points_key}' not found and no fallback Nx3 array in {npz_path}"
                )
            points = fallback

    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"Points must have shape [N, 3], got {points.shape} from {npz_path}"
        )
    if points.shape[0] < 2:
        raise ValueError(f"Point count must be >= 2, got {points.shape[0]} in {npz_path}")

    if num_points is not None:
        points = adjust_num_points(points, num_points, rng)
    if renormalize:
        points = normalize_points(points)
    return points


def chamfer_distance(
    a: np.ndarray,
    b: np.ndarray,
    squared: bool = True,
    kdtree_workers: int = 1,
) -> float:
    tree_a = cKDTree(a)
    tree_b = cKDTree(b)

    dist_a_to_b, _ = tree_b.query(a, k=1, workers=kdtree_workers)
    dist_b_to_a, _ = tree_a.query(b, k=1, workers=kdtree_workers)

    if squared:
        dist_a_to_b = dist_a_to_b ** 2
        dist_b_to_a = dist_b_to_a ** 2

    return float(dist_a_to_b.mean() + dist_b_to_a.mean())


def path_to_shape_id(path: Path, num_points: int | None = None) -> str:
    # Common dataset layout is <shape_id>/model.obj
    if path.name.lower() == "model.obj":
        return path.parent.name
    stem = path.stem
    # Common sample naming from this repo: <source_id>_<num_points>.npz
    if path.suffix.lower() == ".npz":
        parts = stem.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0]
    return stem


def make_unique_ids(paths: List[Path], num_points: int | None = None) -> List[str]:
    counts = {}
    out = []
    for p in paths:
        base = path_to_shape_id(p, num_points=num_points)
        c = counts.get(base, 0)
        counts[base] = c + 1
        if c == 0:
            out.append(base)
        else:
            out.append(f"{base}__{c}")
    return out


def _pair_count_for_range(n: int, start_i: int, end_i: int) -> int:
    count = 0
    for i in range(start_i, end_i):
        count += n - 1 - i
    return count


def _build_pair_tasks(n: int, num_workers: int, tasks_per_worker: int) -> List[Tuple[int, int, int]]:
    if n < 2:
        return []
    num_i = n - 1
    target_tasks = max(1, num_workers * max(1, tasks_per_worker))
    block_size = max(1, math.ceil(num_i / target_tasks))
    tasks: List[Tuple[int, int, int]] = []
    for start_i in range(0, num_i, block_size):
        end_i = min(num_i, start_i + block_size)
        pair_count = _pair_count_for_range(n, start_i, end_i)
        tasks.append((start_i, end_i, pair_count))
    return tasks


def _init_pair_worker(points: List[np.ndarray], squared: bool, kdtree_workers: int):
    global _GLOBAL_POINTS, _GLOBAL_SQUARED, _GLOBAL_KDTREE_WORKERS
    _GLOBAL_POINTS = points
    _GLOBAL_SQUARED = squared
    _GLOBAL_KDTREE_WORKERS = kdtree_workers


def _compute_block_minima(task: Tuple[int, int, int]):
    start_i, end_i, pair_count = task
    points = _GLOBAL_POINTS
    n = len(points)
    local_min_cd = np.full(n, np.inf, dtype=np.float64)
    local_min_idx = np.full(n, -1, dtype=np.int64)

    for i in range(start_i, end_i):
        pi = points[i]
        for j in range(i + 1, n):
            cd = chamfer_distance(
                pi,
                points[j],
                squared=_GLOBAL_SQUARED,
                kdtree_workers=_GLOBAL_KDTREE_WORKERS,
            )
            if cd < local_min_cd[i]:
                local_min_cd[i] = cd
                local_min_idx[i] = j
            if cd < local_min_cd[j]:
                local_min_cd[j] = cd
                local_min_idx[j] = i

    return local_min_cd, local_min_idx, pair_count


def compute_pairwise_min_cd_parallel(
    sampled_points: List[np.ndarray],
    num_workers: int,
    tasks_per_worker: int,
    squared_cd: bool,
    kdtree_workers: int,
):
    n = len(sampled_points)
    min_cd = np.full(n, np.inf, dtype=np.float64)
    min_idx = np.full(n, -1, dtype=np.int64)
    total_pairs = n * (n - 1) // 2

    tasks = _build_pair_tasks(n, num_workers=num_workers, tasks_per_worker=tasks_per_worker)
    print(f"Parallel mode: {num_workers} workers, {len(tasks)} tasks.")

    with mp.Pool(
        processes=num_workers,
        initializer=_init_pair_worker,
        initargs=(sampled_points, squared_cd, kdtree_workers),
    ) as pool:
        with tqdm(total=total_pairs, desc="Pairwise CD") as pbar:
            for local_min_cd, local_min_idx, pair_count in pool.imap_unordered(_compute_block_minima, tasks):
                better = local_min_cd < min_cd
                min_cd[better] = local_min_cd[better]
                min_idx[better] = local_min_idx[better]
                pbar.update(pair_count)

    return min_cd, min_idx


def compute_pairwise_min_cd_serial(
    sampled_points: List[np.ndarray],
    squared_cd: bool,
    kdtree_workers: int,
):
    n = len(sampled_points)
    min_cd = np.full(n, np.inf, dtype=np.float64)
    min_idx = np.full(n, -1, dtype=np.int64)
    total_pairs = n * (n - 1) // 2

    with tqdm(total=total_pairs, desc="Pairwise CD") as pbar:
        for i in range(n - 1):
            for j in range(i + 1, n):
                cd = chamfer_distance(
                    sampled_points[i],
                    sampled_points[j],
                    squared=squared_cd,
                    kdtree_workers=kdtree_workers,
                )
                if cd < min_cd[i]:
                    min_cd[i] = cd
                    min_idx[i] = j
                if cd < min_cd[j]:
                    min_cd[j] = cd
                    min_idx[j] = i
                pbar.update(1)

    return min_cd, min_idx


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute pairwise Chamfer Distance (CD) among OBJ or NPZ shapes, find each shape's "
            "minimum CD and nearest shape ID, and report mean minimum CD."
        )
    )
    parser.add_argument(
        "--input_dirs",
        type=str,
        nargs="+",
        default=None,
        help="One or more directories that contain OBJ files",
    )
    parser.add_argument(
        "--input_npz_dirs",
        type=str,
        nargs="+",
        default=None,
        help="One or more directories that contain NPZ files with sampled points",
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=8192,
        help=(
            "Target number of points. For OBJ: sampled count. "
            "For NPZ: points are resampled to this count unless --keep_npz_points."
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search files in input directories",
    )
    parser.add_argument(
        "--max_shapes",
        type=int,
        default=None,
        help="Optional cap for number of shapes (first N files after sorting)",
    )
    parser.add_argument(
        "--squared_cd",
        action="store_true",
        help="Use squared L2 distances in CD (default: non-squared L2)",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="min_cd_results.csv",
        help="Output CSV path for per-shape nearest-neighbor CD results",
    )
    parser.add_argument(
        "--npz_points_key",
        type=str,
        default="points",
        help="Key to load points from NPZ files (default: points)",
    )
    parser.add_argument(
        "--keep_npz_points",
        action="store_true",
        help="Do not resample NPZ points to --num_points",
    )
    parser.add_argument(
        "--skip_npz_normalize",
        action="store_true",
        help="Skip normalization for NPZ points",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for NPZ point subsampling/upsampling",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of processes for pairwise CD computation (1 = serial)",
    )
    parser.add_argument(
        "--tasks_per_worker",
        type=int,
        default=4,
        help="Task granularity for parallel mode (more tasks => smoother progress updates)",
    )
    parser.add_argument(
        "--kdtree_workers",
        type=int,
        default=1,
        help="Thread workers used inside each KD-tree query (keep 1 for multi-process runs)",
    )
    args = parser.parse_args()

    sampled_points = []
    valid_files = []
    rng = np.random.default_rng(args.seed)

    if not args.input_dirs and not args.input_npz_dirs:
        raise ValueError("Provide at least one of --input_dirs or --input_npz_dirs")

    if args.input_dirs:
        obj_files = gather_files(args.input_dirs, suffix=".obj", recursive=args.recursive)
        if args.max_shapes is not None:
            obj_files = obj_files[: args.max_shapes]
        print(f"Found {len(obj_files)} OBJ files.")
        print("Sampling and normalizing OBJ points...")
        for p in tqdm(obj_files, desc="Load/Sample OBJ"):
            try:
                pts = load_and_sample_points(p, args.num_points)
                sampled_points.append(pts)
                valid_files.append(p)
            except Exception as e:
                print(f"[skip] {p}: {e}")

    if args.input_npz_dirs:
        npz_files = gather_files(args.input_npz_dirs, suffix=".npz", recursive=args.recursive)
        if args.max_shapes is not None:
            remain = max(0, args.max_shapes - len(valid_files))
            npz_files = npz_files[:remain]
        print(f"Found {len(npz_files)} NPZ files.")
        print("Loading NPZ points...")
        target_points = None if args.keep_npz_points else args.num_points
        renormalize = not args.skip_npz_normalize
        for p in tqdm(npz_files, desc="Load NPZ"):
            try:
                pts = load_points_from_npz(
                    p,
                    points_key=args.npz_points_key,
                    num_points=target_points,
                    renormalize=renormalize,
                    rng=rng,
                )
                sampled_points.append(pts)
                valid_files.append(p)
            except Exception as e:
                print(f"[skip] {p}: {e}")

    n = len(sampled_points)
    if n == 0:
        raise RuntimeError("No valid shapes after loading/sampling.")

    shape_ids = make_unique_ids(valid_files, num_points=args.num_points)
    min_cd = np.full(n, np.inf, dtype=np.float64)
    min_idx = np.full(n, -1, dtype=np.int64)

    if n == 1:
        print("Only one valid shape found. No pairwise CD to compute.")
    else:
        total_pairs = n * (n - 1) // 2
        print(f"Computing pairwise CD for {n} shapes ({total_pairs} pairs)...")
        if args.num_workers > 1:
            min_cd, min_idx = compute_pairwise_min_cd_parallel(
                sampled_points,
                num_workers=args.num_workers,
                tasks_per_worker=args.tasks_per_worker,
                squared_cd=args.squared_cd,
                kdtree_workers=args.kdtree_workers,
            )
        else:
            min_cd, min_idx = compute_pairwise_min_cd_serial(
                sampled_points,
                squared_cd=args.squared_cd,
                kdtree_workers=args.kdtree_workers,
            )

    rows = []
    for i in range(n):
        j = int(min_idx[i])
        rows.append(
            {
                "shape_id": shape_ids[i],
                "shape_path": str(valid_files[i]),
                "nearest_shape_id": shape_ids[j] if j >= 0 else "",
                "nearest_shape_path": str(valid_files[j]) if j >= 0 else "",
                "min_cd": float(min_cd[i]) if math.isfinite(min_cd[i]) else np.nan,
            }
        )

    result_df = pd.DataFrame(rows)
    output_csv = Path(args.output_csv).expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_csv, index=False)

    finite_mask = np.isfinite(min_cd)
    mean_min_cd = float(np.mean(min_cd[finite_mask])) if np.any(finite_mask) else float("nan")

    print(f"Saved per-shape nearest-CD results to: {output_csv}")
    print(f"Mean minimum CD across shapes: {mean_min_cd:.8f}")


if __name__ == "__main__":
    main()
