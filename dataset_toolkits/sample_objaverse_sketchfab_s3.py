import argparse
import csv
import os
import re
import tempfile
from dataclasses import dataclass
from urllib.parse import urlparse

import boto3
import numpy as np
import pandas as pd
import trimesh


DEFAULT_S3_PREFIX = (
    "s3://autodesk-adpcdl-965535024567-p-ue1-internal-build3d-meshes/"
    "private/unstructured/large-dataset-meshes-stl-ue1/objaverse"
)


@dataclass
class InstanceRecord:
    sha256: str
    file_identifier: str
    sketchfab_id: str


def parse_sketchfab_id(file_identifier: str) -> str:
    """
    Extract the 32-char Sketchfab model id from a URL like:
    https://sketchfab.com/3d-models/<slug-or-id>
    """
    if not isinstance(file_identifier, str) or not file_identifier:
        raise ValueError(f"Invalid file_identifier: {file_identifier!r}")

    parsed = urlparse(file_identifier)
    path_parts = [x for x in parsed.path.split("/") if x]
    if not path_parts:
        raise ValueError(f"Cannot parse file_identifier path: {file_identifier}")

    last_part = path_parts[-1]
    # Common case: 32-char hex id, either the whole segment or at segment end.
    m = re.search(r"([0-9a-fA-F]{32})$", last_part)
    if m:
        return m.group(1).lower()
    m = re.search(r"([0-9a-fA-F]{32})", last_part)
    if m:
        return m.group(1).lower()

    # Fallback: some metadata entries are non-hex sketchfab ids.
    # If URL is like "<slug>-<id>", use the final token; otherwise use full segment.
    if "-" in last_part:
        tail = last_part.rsplit("-", 1)[-1].strip()
        if tail:
            return tail
    return last_part.strip()


def parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got: {s3_uri}")
    no_prefix = s3_uri[5:]
    bucket, _, key = no_prefix.partition("/")
    if not bucket or not key:
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    return bucket, key


def load_instance_from_metadata(
    metadata_csv: str,
    sha256: str | None,
    file_identifier: str | None,
    sketchfab_id: str | None,
) -> InstanceRecord:
    metadata = pd.read_csv(metadata_csv, usecols=["sha256", "file_identifier"])

    if sha256 is not None:
        rows = metadata[metadata["sha256"] == sha256]
        if rows.empty:
            raise ValueError(f"sha256 not found in metadata: {sha256}")
        row = rows.iloc[0]
        fid = str(row["file_identifier"])
        return InstanceRecord(
            sha256=str(row["sha256"]),
            file_identifier=fid,
            sketchfab_id=parse_sketchfab_id(fid),
        )

    if file_identifier is not None:
        rows = metadata[metadata["file_identifier"] == file_identifier]
        if rows.empty:
            raise ValueError(f"file_identifier not found in metadata: {file_identifier}")
        row = rows.iloc[0]
        fid = str(row["file_identifier"])
        return InstanceRecord(
            sha256=str(row["sha256"]),
            file_identifier=fid,
            sketchfab_id=parse_sketchfab_id(fid),
        )

    if sketchfab_id is not None:
        sketchfab_id = sketchfab_id.lower()
        ids = metadata["file_identifier"].astype(str).map(parse_sketchfab_id)
        rows = metadata[ids == sketchfab_id]
        if rows.empty:
            raise ValueError(f"sketchfab_id not found in metadata: {sketchfab_id}")
        row = rows.iloc[0]
        fid = str(row["file_identifier"])
        return InstanceRecord(
            sha256=str(row["sha256"]),
            file_identifier=fid,
            sketchfab_id=sketchfab_id,
        )

    row = metadata.iloc[0]
    fid = str(row["file_identifier"])
    return InstanceRecord(
        sha256=str(row["sha256"]),
        file_identifier=fid,
        sketchfab_id=parse_sketchfab_id(fid),
    )


def load_metadata_with_sketchfab_id(metadata_csv: str) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_csv, usecols=["sha256", "file_identifier"]).copy()
    metadata["sha256"] = metadata["sha256"].astype(str)
    metadata["file_identifier"] = metadata["file_identifier"].astype(str)
    metadata["sketchfab_id"] = metadata["file_identifier"].map(parse_sketchfab_id)
    return metadata


def resolve_instance_from_df(
    metadata: pd.DataFrame,
    id_field: str,
    id_value: str,
) -> InstanceRecord | None:
    if id_field == "sha256":
        rows = metadata[metadata["sha256"] == id_value]
    elif id_field == "file_identifier":
        rows = metadata[metadata["file_identifier"] == id_value]
    elif id_field == "sketchfab_id":
        rows = metadata[metadata["sketchfab_id"] == id_value]
    else:
        raise ValueError(f"Unsupported id_field: {id_field}")

    if rows.empty:
        return None
    row = rows.iloc[0]
    return InstanceRecord(
        sha256=str(row["sha256"]),
        file_identifier=str(row["file_identifier"]),
        sketchfab_id=str(row["sketchfab_id"]),
    )


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


def sample_points_from_obj(
    s3_uri: str,
    num_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    bucket, key = parse_s3_uri(s3_uri)
    s3 = boto3.client("s3")

    with tempfile.TemporaryDirectory() as tmp_dir:
        local_obj = os.path.join(tmp_dir, "model.obj")
        s3.download_file(bucket, key, local_obj)

        loaded = trimesh.load(local_obj, force="scene", process=False)
        mesh = scene_to_mesh(loaded)
        points, face_idx = trimesh.sample.sample_surface(mesh, num_points)
        normals = mesh.face_normals[face_idx]
        return points.astype(np.float32), normals.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Read ObjaverseXL_sketchfab metadata, extract sketchfab id from "
            "file_identifier, fetch model.obj from S3, and sample points."
        )
    )
    parser.add_argument(
        "--metadata_csv",
        type=str,
        default="../datasets/ObjaverseXL_sketchfab/metadata.csv",
        help="Path to ObjaverseXL_sketchfab metadata.csv",
    )
    parser.add_argument(
        "--s3_prefix",
        type=str,
        default=DEFAULT_S3_PREFIX,
        help="S3 prefix ending at .../objaverse",
    )
    parser.add_argument(
        "--sha256",
        type=str,
        default=None,
        help="Select instance by sha256",
    )
    parser.add_argument(
        "--file_identifier",
        type=str,
        default=None,
        help="Select instance by file_identifier URL",
    )
    parser.add_argument(
        "--sketchfab_id",
        type=str,
        default=None,
        help="Select instance by sketchfab id (32 hex chars)",
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=8192,
        help="Number of points to sample from mesh surface",
    )
    parser.add_argument(
        "--output_npz",
        type=str,
        default=None,
        help="Output .npz path; default: ../datasets/ObjaverseXL_sketchfab/samples/<id>_<N>.npz",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only resolve metadata row + S3 URI, do not download/sample",
    )
    parser.add_argument(
        "--save_ply",
        action="store_true",
        help="Also save sampled points as PLY next to output NPZ",
    )
    parser.add_argument(
        "--id_list",
        type=str,
        default=None,
        help=(
            "Optional text file with one id per line for batch processing. "
            "If set, script runs in batch mode."
        ),
    )
    parser.add_argument(
        "--id_field",
        type=str,
        choices=["sha256", "sketchfab_id", "file_identifier"],
        default="sketchfab_id",
        help="Which field --id_list lines represent",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Batch output directory for NPZ/PLY; default: <metadata_dir>/samples",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="In batch mode, overwrite existing outputs (default is skip existing)",
    )
    parser.add_argument(
        "--batch_status_csv",
        type=str,
        default=None,
        help="Optional CSV to save per-id status in batch mode",
    )
    args = parser.parse_args()

    if args.id_list is not None:
        metadata = load_metadata_with_sketchfab_id(args.metadata_csv)
        if args.output_dir is None:
            out_dir = os.path.normpath(
                os.path.join(os.path.dirname(args.metadata_csv), "samples")
            )
        else:
            out_dir = args.output_dir
        os.makedirs(out_dir, exist_ok=True)

        with open(args.id_list, "r") as f:
            ids = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

        status_rows = []
        for idx, raw_id in enumerate(ids, start=1):
            record = resolve_instance_from_df(metadata, args.id_field, raw_id)
            if record is None:
                print(f"[{idx}/{len(ids)}] skip_not_in_metadata: {raw_id}")
                status_rows.append(
                    {"id": raw_id, "status": "not_in_metadata", "output_npz": "", "error": ""}
                )
                continue

            s3_uri = f"{args.s3_prefix.rstrip('/')}/{record.sketchfab_id}/model.obj"
            output_npz = os.path.join(out_dir, f"{record.sketchfab_id}_{args.num_points}.npz")

            if os.path.exists(output_npz) and not args.overwrite:
                print(f"[{idx}/{len(ids)}] skip_exists: {output_npz}")
                status_rows.append(
                    {"id": raw_id, "status": "skip_exists", "output_npz": output_npz, "error": ""}
                )
                continue

            if args.dry_run:
                print(
                    f"[{idx}/{len(ids)}] dry_run: id={raw_id} "
                    f"sketchfab_id={record.sketchfab_id} s3_uri={s3_uri}"
                )
                status_rows.append(
                    {"id": raw_id, "status": "dry_run", "output_npz": output_npz, "error": ""}
                )
                continue

            try:
                points, normals = sample_points_from_obj(
                    s3_uri=s3_uri,
                    num_points=args.num_points,
                )
                np.savez_compressed(
                    output_npz,
                    points=points,
                    normals=normals,
                    sha256=record.sha256,
                    file_identifier=record.file_identifier,
                    sketchfab_id=record.sketchfab_id,
                    s3_uri=s3_uri,
                )
                if args.save_ply:
                    ply_path = os.path.splitext(output_npz)[0] + ".ply"
                    cloud = trimesh.points.PointCloud(points)
                    cloud.export(ply_path)
                print(f"[{idx}/{len(ids)}] done: {output_npz}")
                status_rows.append(
                    {"id": raw_id, "status": "done", "output_npz": output_npz, "error": ""}
                )
            except Exception as e:
                print(f"[{idx}/{len(ids)}] skip_error: id={raw_id} err={e}")
                status_rows.append(
                    {"id": raw_id, "status": "error", "output_npz": output_npz, "error": str(e)}
                )

        if args.batch_status_csv is None:
            status_csv = os.path.join(out_dir, "batch_status.csv")
        else:
            status_csv = args.batch_status_csv
        os.makedirs(os.path.dirname(os.path.abspath(status_csv)), exist_ok=True)
        with open(status_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "status", "output_npz", "error"])
            writer.writeheader()
            writer.writerows(status_rows)

        total = len(status_rows)
        done = sum(1 for r in status_rows if r["status"] == "done")
        dry = sum(1 for r in status_rows if r["status"] == "dry_run")
        miss = sum(1 for r in status_rows if r["status"] == "not_in_metadata")
        exist = sum(1 for r in status_rows if r["status"] == "skip_exists")
        err = sum(1 for r in status_rows if r["status"] == "error")
        print(
            f"batch_summary total={total} done={done} dry_run={dry} "
            f"skip_exists={exist} not_in_metadata={miss} error={err}"
        )
        print(f"status_csv: {status_csv}")
        return

    instance = load_instance_from_metadata(
        metadata_csv=args.metadata_csv,
        sha256=args.sha256,
        file_identifier=args.file_identifier,
        sketchfab_id=args.sketchfab_id,
    )

    s3_uri = f"{args.s3_prefix.rstrip('/')}/{instance.sketchfab_id}/model.obj"

    print(f"sha256: {instance.sha256}")
    print(f"file_identifier: {instance.file_identifier}")
    print(f"sketchfab_id: {instance.sketchfab_id}")
    print(f"s3_uri: {s3_uri}")

    if args.dry_run:
        return

    points, normals = sample_points_from_obj(s3_uri=s3_uri, num_points=args.num_points)

    if args.output_npz is None:
        out_dir = os.path.normpath(
            os.path.join(os.path.dirname(args.metadata_csv), "samples")
        )
        os.makedirs(out_dir, exist_ok=True)
        output_npz = os.path.join(
            out_dir, f"{instance.sketchfab_id}_{args.num_points}.npz"
        )
    else:
        output_npz = args.output_npz
        os.makedirs(os.path.dirname(os.path.abspath(output_npz)), exist_ok=True)

    np.savez_compressed(
        output_npz,
        points=points,
        normals=normals,
        sha256=instance.sha256,
        file_identifier=instance.file_identifier,
        sketchfab_id=instance.sketchfab_id,
        s3_uri=s3_uri,
    )

    print(f"saved_npz: {output_npz}")

    if args.save_ply:
        ply_path = os.path.splitext(output_npz)[0] + ".ply"
        cloud = trimesh.points.PointCloud(points)
        cloud.export(ply_path)
        print(f"saved_ply: {ply_path}")


if __name__ == "__main__":
    main()
