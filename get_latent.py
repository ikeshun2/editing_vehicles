#!/usr/bin/env python3
# scripts/infer_latents_from_sdf_new.py
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import random
import pickle

from experimet_config import Config

# --- user config ---
DATA_ROOT = "data_test/sdf"                     # フォルダ直下に各車フォルダがある想定
DEEPSDF_WEIGHTS = "./outputs/models/deepsdf_best_epoch4000.pth"
OUT_SUMMARY = os.path.join("outputs/latent-codes", "latent_codes_new.pth")
OUT_ARRAY = os.path.join("outputs/latent-codes", "latent_codes_new_array.pth")
OUT_MAP_PICKLE_PREFIX = os.path.join("outputs/data-split", "carlatent_new")  # will append _carfolder2latent_idx.pickle etc
SAVE_PER_FOLDER = True                       # 各フォルダ直下に latent.pth を保存するか
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- optimization config ---
class Cfg:
    latent_code_dim = 128    # 実際の DeepSDF の latent dim に合わせてください
    optimize_step = 2000
    lr = 1e-3                # 初期学習率 (1e-3)
    init_scale = 1e-2
    l2_prior_lambda = 1e-4  #正則化項目(1e-3が最適？)
    num_restarts = 1
    print_every = 200
    max_points = 10000    # None -> 全点。プールとして保持する最大点数
    seed = 42

cfg = Cfg()

# --- helpers ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(cfg.seed)

# import DeepSDF
from src.models.deepsdf import DeepSDF

def build_deepsdf(device):
    model = DeepSDF(
        hidden_dim=512 if not hasattr(cfg, "hidden_dim") else cfg.hidden_dim,
        xyz_pos_enc_dim=3 if not hasattr(cfg, "xyz_pos_enc_dim") else cfg.xyz_pos_enc_dim,
        latent_code_dim=cfg.latent_code_dim,
        dropout_prob=0.0
    ).to(device)
    state = torch.load(DEEPSDF_WEIGHTS, map_location=device)
    model.load_state_dict(state)
    model.eval()
    # freeze params
    for p in model.parameters():
        p.requires_grad = False
    return model

def optimize_latent_for_shape(deep_sdf, xyz, sdf_target, cfg, device):
    loss_fn = nn.MSELoss(reduction="mean")
    best_overall_latent = None
    best_overall_loss = float("inf")

    # サンプリング処理
    N = xyz.size(0)
    if cfg.max_points is not None and N > cfg.max_points:
        rng = torch.Generator(device=device).manual_seed(cfg.seed)
        idx = torch.randperm(N, generator=rng, device=device)[: cfg.max_points]
        xyz_sub = xyz[idx]
        sdf_sub = sdf_target[idx]
    else:
        xyz_sub = xyz
        sdf_sub = sdf_target

    # ミニバッチのサイズ
    batch_size = 30000
    sub_N = xyz_sub.size(0)

    for restart in range(cfg.num_restarts):
        latent = nn.Parameter(torch.randn(1, cfg.latent_code_dim, device=device) * cfg.init_scale)
        optimizer = torch.optim.Adam([latent], lr=cfg.lr)
        
        # ==========================================
        # ★ 追加：400ステップ目で学習率に0.1を掛ける（1e-3 -> 1e-4）
        # ==========================================
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[400], gamma=0.1)
        
        best_local_loss = float("inf")
        best_local_latent = latent.detach().clone()

        pbar = tqdm(range(1, cfg.optimize_step + 1), desc=f"  Restart {restart+1}/{cfg.num_restarts}", leave=False)
        
        for step in pbar:
            optimizer.zero_grad()

            # プールからバッチサイズ分だけランダム抽出
            batch_idx = torch.randint(0, sub_N, (batch_size,), device=device)
            step_xyz = xyz_sub[batch_idx]
            step_sdf = sdf_sub[batch_idx]

            pred = deep_sdf(latent, step_xyz).view(-1)
            target = step_sdf.view(-1)
            
            loss_data = loss_fn(pred, target)
            loss_prior = cfg.l2_prior_lambda * torch.sum(latent**2)
            loss = loss_data + loss_prior
            
            loss.backward()
            optimizer.step()
            
            # ==========================================
            # ★ 追加：スケジューラーを1歩進める
            # ==========================================
            scheduler.step()

            current_loss = loss.item()
            if current_loss < best_local_loss:
                best_local_loss = current_loss
                best_local_latent = latent.detach().clone()

            # プログレスバーの更新
            if step % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    "loss": f"{current_loss:.4e}",
                    "best": f"{best_local_loss:.4e}",
                    "lr": f"{current_lr:.0e}"  # 現在の学習率を表示
                })

        print(f"  > Restart {restart+1} finished. Best Loss: {best_local_loss:.6e}")
        
        if best_local_loss < best_overall_loss:
            best_overall_loss = best_local_loss
            best_overall_latent = best_local_latent.clone()

    print(f"  >> Optimized Result: Overall Best Loss = {best_overall_loss:.6e}")
    return best_overall_latent

def load_points_and_sdf(folder_path):
    pts_path = os.path.join(folder_path, "points.npy")
    sdf_path = os.path.join(folder_path, "sdf.npy")
    if not (os.path.exists(pts_path) and os.path.exists(sdf_path)):
        raise FileNotFoundError(f"missing points/sdf in {folder_path}")
    pts = np.load(pts_path).astype(np.float32)    # shape (N,3)
    sdf = np.load(sdf_path).astype(np.float32)    # shape (N,) or (N,1)
    return pts, sdf

def main():
    os.makedirs(os.path.dirname(OUT_SUMMARY), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_ARRAY), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_MAP_PICKLE_PREFIX), exist_ok=True)

    deep_sdf = build_deepsdf(DEVICE)
    folders = sorted([d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))])
    latent_dict = {}

    for folder in tqdm(folders, desc="folders"):
        folder_path = os.path.join(DATA_ROOT, folder)
        try:
            pts_np, sdf_np = load_points_and_sdf(folder_path)
        except Exception as e:
            print(f" skip {folder}: {e}")
            continue

        # convert to torch
        xyz = torch.from_numpy(pts_np).float().to(DEVICE)            # (N,3)
        sdf = torch.from_numpy(sdf_np).float().to(DEVICE)
        # if sdf shape is (N,1) flatten it
        if sdf.dim() > 1 and sdf.size(1) == 1:
            sdf = sdf.view(-1)

        print(f"\nProcessing {folder} | points: {xyz.shape[0]}")
        best_latent = optimize_latent_for_shape(deep_sdf, xyz, sdf, cfg, DEVICE)  # tensor shape (1,D)
        if best_latent is None:
            print(f" failed for {folder}")
            continue

        # save per-folder
        if SAVE_PER_FOLDER:
            save_path = os.path.join(folder_path, "latent.pth")
            torch.save(best_latent.detach().cpu(), save_path)

        latent_dict[folder] = best_latent.detach().cpu()  # store on cpu

    # save summary dictionary (folder -> tensor(1,D))
    torch.save(latent_dict, OUT_SUMMARY)
    print(f"\nSaved summary latents (dict) to: {OUT_SUMMARY}")

    # build array and mapping for compatibility with existing scripts
    folders_sorted = sorted(latent_dict.keys())
    latent_list = [latent_dict[f].view(-1) for f in folders_sorted]  # each (D,)
    if len(latent_list) == 0:
        print("No latents found; exiting.")
        return
    latent_tensor = torch.stack(latent_list, dim=0)  # (N, D)
    torch.save(latent_tensor, OUT_ARRAY)
    print(f"Saved latent array to: {OUT_ARRAY} shape: {latent_tensor.shape}")

    carfolder2latent_idx = {f: i for i, f in enumerate(folders_sorted)}
    latent_idx2carfolder = {i: f for i, f in enumerate(folders_sorted)}

    with open(OUT_MAP_PICKLE_PREFIX + "_carfolder2latent_idx.pickle", "wb") as f:
        pickle.dump(carfolder2latent_idx, f)
    with open(OUT_MAP_PICKLE_PREFIX + "_latent_idx2carfolder.pickle", "wb") as f:
        pickle.dump(latent_idx2carfolder, f)
    print(f"Saved mapping pickles with prefix: {OUT_MAP_PICKLE_PREFIX}_*.pickle")

if __name__ == "__main__":
    main()
