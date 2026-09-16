#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
stl2sdf-good.py

STL -> points + SDF sampling pipeline optimized for car shapes:
- reads all .stl files under <data_dir>/stl/**
- uses mesh_to_sdf.sample_sdf_near_surface to sample many points near the surface
  and fewer points far from the surface (good for automotive surfaces)
- saves outputs to <data_dir>/sdf/<basename>/{points.npy, sdf.npy, meta.npy}

Usage:
    python stl2sdf-good.py
    (You can now simply run `python stl2sdf-good.py` to use default settings)
    xvfb-run -a python3 stl2sdf-good.py --data_dir ./data --overwriteを使えばいい
"""

import os
import sys
import argparse
import json
import random
from glob import glob
from pathlib import Path

import numpy as np
import trimesh
from tqdm.auto import tqdm

from mesh_to_sdf import sample_sdf_near_surface


def find_stl_files(data_dir: str):
    # data/stl の直下にある STL ファイルのみを対象にする
    pattern = os.path.join(data_dir, "stl", "*.stl")
    files = glob(pattern)
    return sorted(files)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def process_mesh(stl_path: str,
                 out_dir: str,
                 number_of_points: int,
                 surface_point_method: str,
                 sign_method: str,
                 scan_count: int,
                 scan_resolution: int,
                 sample_point_count: int,
                 normal_sample_count: int,
                 overwrite: bool = False):
    base_name = Path(stl_path).stem
    mesh_out_dir = os.path.join(out_dir, base_name)
    ensure_dir(mesh_out_dir)

    points_path = os.path.join(mesh_out_dir, "points.npy")
    sdf_path = os.path.join(mesh_out_dir, "sdf.npy")
    meta_path = os.path.join(mesh_out_dir, "meta.json")

    if not overwrite and os.path.exists(points_path) and os.path.exists(sdf_path):
        return "skipped", mesh_out_dir

    # load mesh
    mesh = trimesh.load(stl_path, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh) and hasattr(mesh, 'geometry'):
        # some STL files may load as Scene with single geometry
        # try to merge
        try:
            mesh = trimesh.util.concatenate([g for g in mesh.geometry.values()])
        except Exception:
            mesh = mesh.dump(concatenate=True)

    # call sampler
    try:
        points, sdf = sample_sdf_near_surface(
            mesh,
            number_of_points=number_of_points,
            surface_point_method=surface_point_method,
            sign_method=sign_method,
            scan_count=scan_count,
            scan_resolution=scan_resolution,
            sample_point_count=sample_point_count,
            normal_sample_count=normal_sample_count,
            min_size=0,
            return_gradients=False,
        )
    except Exception as e:
        # try fallback with lighter settings
        try:
            print(f"  Warning: sampling failed for {stl_path} with error {e}. Retrying with lighter settings...")
            points, sdf = sample_sdf_near_surface(
                mesh,
                number_of_points=max(100000, number_of_points // 4),
                surface_point_method=surface_point_method,
                sign_method=sign_method,
                scan_count=max(10, scan_count // 2),
                scan_resolution=max(64, scan_resolution // 2),
                sample_point_count=max(200000, sample_point_count // 10),
                normal_sample_count=max(8, normal_sample_count // 4),
                min_size=0,
                return_gradients=False,
            )
        except Exception as e2:
            return f"error: {e2}", mesh_out_dir

    # save
    np.save(points_path, points)
    np.save(sdf_path, sdf)

    meta = {
        "source_stl": os.path.abspath(stl_path),
        "num_points": int(points.shape[0]),
        "sdf_shape": sdf.shape,
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    return "ok", mesh_out_dir


def main():
    # コマンドライン引数の省略設定
    parser = argparse.ArgumentParser(description="Convert STL to SDF with default options.")
    parser.add_argument("--data_dir", type=str, default=os.getcwd(), help="base data dir which contains stl/ and sdf/ (default: current directory)")
    parser.add_argument("--number_of_points", type=int, default=2000000, help="total number of sample points to generate per mesh")
    parser.add_argument("--surface_point_method", type=str, default="scan", help="surface sampling method passed to mesh_to_sdf (e.g. 'scan')")
    parser.add_argument("--sign_method", type=str, default="normal", help="sign method for inside/outside detection")
    parser.add_argument("--scan_count", type=int, default=100, help="scan_count for sample_sdf_near_surface")
    parser.add_argument("--scan_resolution", type=int, default=400, help="scan_resolution for sample_sdf_near_surface")
    parser.add_argument("--sample_point_count", type=int, default=5000000, help="internal sample_point_count (controls density far from surface)")
    parser.add_argument("--normal_sample_count", type=int, default=200, help="normal_sample_count")
    parser.add_argument("--shuffle", action="store_true", help="shuffle file list before processing")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing outputs")
    parser.add_argument("--seed", type=int, default=0, help="random seed for shuffling")

    # デフォルト引数を用意
    args = parser.parse_args()

    # STLファイルのあるディレクトリと出力先を決定
    data_dir = args.data_dir
    stl_dir = os.path.join(data_dir, "stl")
    out_root = os.path.join(data_dir, "sdf")
    ensure_dir(out_root)

    files = find_stl_files(data_dir)
    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(files)

    error_files = []
    success = []

    if len(files) == 0:
        print(f"No .stl files found under {stl_dir}")
        sys.exit(1)

    for stl_path in tqdm(files, desc="STL -> SDF"):
        try:
            status, mesh_out = process_mesh(
                stl_path,
                out_root,
                number_of_points=args.number_of_points,
                surface_point_method=args.surface_point_method,
                sign_method=args.sign_method,
                scan_count=args.scan_count,
                scan_resolution=args.scan_resolution,
                sample_point_count=args.sample_point_count,
                normal_sample_count=args.normal_sample_count,
                overwrite=args.overwrite,
            )
            if status != "ok":
                error_files.append((stl_path, status))
            else:
                success.append(mesh_out)
        except Exception as e:
            error_files.append((stl_path, str(e)))

    print("\nSummary:")
    print(f"  processed: {len(success)}")
    print(f"  errors: {len(error_files)}")
    if len(error_files) > 0:
        for ef, reason in error_files:
            print(f"    - {ef}: {reason}")


if __name__ == "__main__":
    main()

