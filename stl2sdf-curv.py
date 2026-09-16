#!/usr/bin/env python
# -*- coding: utf-8 -*-
# これは，stl2sdf-good.pyの改変バージョンである．形状のより高曲率な部分に着目してSDF計算点をサンプリングする．
"""
stl2sdf-curv.py

STL -> points + SDF sampling pipeline optimized for car shapes:
- reads all .stl files under ./data/stl/
- Normalizes the mesh to a [-1, 1] unit sphere first.
- uses mesh_to_sdf.sample_sdf_near_surface for baseline sampling (e.g. 2,000,000 points)
- Calculates face curvatures and performs targeted sampling on high-curvature 
  areas (e.g., additional 500,000 points) for detailed automotive features.
- saves outputs to ./data/sdf/<basename>/{points.npy, sdf.npy, meta.json}

Usage:
    python stl2sdf-curv.py
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

# sample_sdf_near_surface に加えて、任意の空間座標のSDFを計算する mesh_to_sdf もインポート
from mesh_to_sdf import sample_sdf_near_surface, mesh_to_sdf


def find_stl_files(data_dir: str):
    # data/stl の直下にある STL ファイルのみを対象にする (相対パス)
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
                 extra_points_count: int,
                 noise_scale_factor: float,
                 overwrite: bool = False):
    
    # ファイル名（拡張子なし）を取得（例: "car_001.stl" -> "car_001"）
    base_name = Path(stl_path).stem
    # 保存先のサブディレクトリを作成（例: "data/sdf/car_001"）
    mesh_out_dir = os.path.join(out_dir, base_name)
    ensure_dir(mesh_out_dir)

    # 格納するファイルのパスを設定
    points_path = os.path.join(mesh_out_dir, "points.npy")
    sdf_path = os.path.join(mesh_out_dir, "sdf.npy")
    meta_path = os.path.join(mesh_out_dir, "meta.json")

    if not overwrite and os.path.exists(points_path) and os.path.exists(sdf_path):
        return "skipped", mesh_out_dir

    # 1. Load mesh
    mesh = trimesh.load(stl_path, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh) and hasattr(mesh, 'geometry'):
        try:
            mesh = trimesh.util.concatenate([g for g in mesh.geometry.values()])
        except Exception:
            mesh = mesh.dump(concatenate=True)

    # =====================================================================
    # 【追加：メッシュの正規化】
    # ここで先にメッシュを [-1, 1] の単位球に収まるように正規化します。
    # これにより、ベースラインと追加サンプリングの座標系が完全に一致します。
    # =====================================================================
    # ① バウンディングボックスの中心を原点 (0,0,0) に移動
    centroid = mesh.bounds.mean(axis=0)
    mesh.vertices -= centroid
    
    # ② 原点から最も遠い頂点の距離を計算し、最大半径が1になるように全体を縮小
    max_distance = np.max(np.linalg.norm(mesh.vertices, axis=1))
    if max_distance > 0:
        mesh.vertices /= max_distance
    # =====================================================================

    # 2. Baseline Sampling (2,000,000 points by default)
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
        # Fallback for baseline sampling
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

    # 3. High-Curvature Targeted Sampling (500,000 points by default)
    if extra_points_count > 0:
        try:
            # 各面の「曲率スコア」を計算
            face_curvatures = np.zeros(len(mesh.faces))
            if len(mesh.face_adjacency) > 0:
                np.add.at(face_curvatures, mesh.face_adjacency[:, 0], mesh.face_adjacency_angles)
                np.add.at(face_curvatures, mesh.face_adjacency[:, 1], mesh.face_adjacency_angles)

            # 平坦な面への「参加賞」と、面積による重み付けを廃止（純粋な曲率スコアだけを使う）
            face_weights = face_curvatures
            weight_sum = np.sum(face_weights)
            
            if weight_sum > 0 and not np.isnan(weight_sum):
                face_probabilities = face_weights / weight_sum
            else:
                # すべてが完全に真っ平ら（あり得ないが念のため）の場合は均等にする
                face_probabilities = np.ones(len(mesh.faces)) / len(mesh.faces)

            # 確率に基づいて面をサンプリング
            chosen_faces = np.random.choice(len(mesh.faces), size=extra_points_count, p=face_probabilities)

            # 選ばれた面（三角形）の内部でランダムな表面点を生成（重心座標系）
            triangles = mesh.triangles[chosen_faces]
            u = np.random.rand(extra_points_count, 1)
            v = np.random.rand(extra_points_count, 1)
            is_outside = u + v > 1
            u[is_outside] = 1 - u[is_outside]
            v[is_outside] = 1 - v[is_outside]
            w = 1 - (u + v)

            # 表面上の追加ポイント座標を計算
            surface_points = (u * triangles[:, 0, :] + v * triangles[:, 1, :] + w * triangles[:, 2, :])

            # 入り組んだパーツでのSDF破綻を防ぐための微小なノイズスケール（デフォルト 0.5%）
            bbox_extent = np.max(mesh.bounds[1] - mesh.bounds[0])
            noise_scale = bbox_extent * noise_scale_factor
            
            # 正規分布ノイズを加えて空間点に摂動
            perturbation = np.random.normal(scale=noise_scale, size=surface_points.shape)
            extra_query_points = surface_points + perturbation

            # 追加点に対するSDFを計算
            extra_sdf = mesh_to_sdf(
                mesh, 
                extra_query_points, 
                surface_point_method=surface_point_method, 
                sign_method=sign_method
            )

            # 既存のポイントと結合
            points = np.vstack((points, extra_query_points))
            sdf = np.concatenate((sdf, extra_sdf))
            
        except Exception as e:
            print(f"  Warning: Curvature targeted sampling failed for {stl_path} with error {e}. Skipping extra points.")

    # 4. Save results
    np.save(points_path, points)
    np.save(sdf_path, sdf)

    meta = {
        "source_stl": stl_path,  # 相対パスのまま記録
        "num_points": int(points.shape[0]),
        "sdf_shape": sdf.shape,
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    return "ok", mesh_out_dir


def main():
    parser = argparse.ArgumentParser(description="Convert STL to SDF with curvature-based dense sampling.")
    
    parser.add_argument("--data_dir", type=str, default="./data", help="base data dir which contains stl/ and sdf/ (default: ./data)")
    
    # ベースラインサンプリングのパラメータ
    parser.add_argument("--number_of_points", type=int, default=2000000, help="total number of baseline sample points to generate per mesh")
    parser.add_argument("--surface_point_method", type=str, default="scan", help="surface sampling method passed to mesh_to_sdf (e.g. 'scan')")
    parser.add_argument("--sign_method", type=str, default="normal", help="sign method for inside/outside detection")
    parser.add_argument("--scan_count", type=int, default=100, help="scan_count for sample_sdf_near_surface")
    parser.add_argument("--scan_resolution", type=int, default=400, help="scan_resolution for sample_sdf_near_surface")
    parser.add_argument("--sample_point_count", type=int, default=5000000, help="internal sample_point_count (controls density far from surface)")
    parser.add_argument("--normal_sample_count", type=int, default=200, help="normal_sample_count")
    
    # 曲率に基づく追加サンプリングのパラメータ
    parser.add_argument("--extra_points_count", type=int, default=500000, help="number of extra points sampled near high curvature areas")
    # デフォルトのノイズスケールを 0.5% (0.005) に縮小
    parser.add_argument("--noise_scale_factor", type=float, default=0.005, help="noise scale as a fraction of the bounding box size (default 0.5%)")

    # その他の設定
    parser.add_argument("--shuffle", action="store_true", help="shuffle file list before processing")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing outputs")
    parser.add_argument("--seed", type=int, default=0, help="random seed for shuffling")

    args = parser.parse_args()

    # ディレクトリパスの組み立て
    data_dir = args.data_dir
    stl_dir = os.path.join(data_dir, "stl")   # => ./data/stl
    out_root = os.path.join(data_dir, "sdf")  # => ./data/sdf
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
                extra_points_count=args.extra_points_count,
                noise_scale_factor=args.noise_scale_factor,
                overwrite=args.overwrite,
            )
            if status != "ok":
                error_files.append((stl_path, status))
            else:
                success.append(mesh_out)
        except Exception as e:
            error_files.append((stl_path, str(e)))

    # ログファイルへの書き出し処理
    print("\nSummary:")
    print(f"  processed: {len(success)}")
    print(f"  errors: {len(error_files)}")
    if len(error_files) > 0:
        for ef, reason in error_files:
            print(f"    - {ef}: {reason}")

    log_path = os.path.join(data_dir, "process_summary.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("Summary:\n")
        f.write(f"  processed: {len(success)}\n")
        f.write(f"  errors: {len(error_files)}\n")
        if len(error_files) > 0:
            for ef, reason in error_files:
                f.write(f"    - {ef}: {reason}\n")
    print(f"\n※ログを {log_path} に保存しました。")


if __name__ == "__main__":
    main()

