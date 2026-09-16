#!/usr/bin/env python3
"""
apply_cd_edit_and_export_new_latent.py

用途:
  - infer_latents_from_sdf_new.py で生成した latent_codes_new_array.pth を読み込む
  - latent に対して ΔCd 編集を適用
  - 編集前/編集後の PLY を出力
  - regressor による Cd 推定値を表示
  - ΔCd も表示
  - 編集前後の real Cd 値、および
    設計変数(29)・意匠性(14)・意匠性主成分(3)の変化量(Delta)を CSV に保存する
"""

import os
import random
import pickle
import time

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

# project path
import sys
sys.path.append("./")

from experimet_config import Config
from src.models.deepsdf import DeepSDF
from src.models.latent_walker import WalkMlpMultiW
from src.utils.mesh_utils import create_mesh

# optional regressor
USE_REGRESSOR_FOR_REPORT = True
try:
    from src.models.regressor import Regressor
except Exception:
    USE_REGRESSOR_FOR_REPORT = False


# =========================================================
# User Config
# =========================================================

TARGET_DELTA_CD = -0.01
NUM_STEPS = 1

OUTPUT_DIR = "./outputs/mesh/edited_before_after_newlatent"
# ★ 出力先を how_change ディレクトリに変更
OUTPUT_CSV_DIR = "./outputs/how_change"
OUTPUT_CSV_PATH = os.path.join(OUTPUT_CSV_DIR, "how_change.csv")

DEVICE_OVERRIDE = None
SAMPLE_NUM = 20
APPLY_METHOD = "scalar"   # or "delta_via_regressor"

SEED = 42
PRINT_LATENT_HEAD = True
ONLY_FIRST_N = None

# 0: before形状を出力しない, 1: before形状を出力する
EXPORT_BEFORE_FLAG = 0

# 通常create_mesh用解像度
MESH_RESOLUTION = 500

# =========================================================


if SEED is not None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

cfg = Config()

if DEVICE_OVERRIDE:
    cfg.device = DEVICE_OVERRIDE

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)


# =========================================================
# helper
# =========================================================

def scale_cd_change(
    delta_cd,
    merge_csv_path="./data/table/all_data.csv",
    noise_data=[]
):
    if not os.path.exists(merge_csv_path):
        raise FileNotFoundError(f"csv not found: {merge_csv_path}")

    df = pd.read_csv(merge_csv_path).dropna().reset_index(drop=True)

    if noise_data:
        df = df[~df["folder_name"].isin(noise_data)]

    if "Cd" in df.columns:
        min_cd = df["Cd"].min()
        max_cd = df["Cd"].max()
    elif "Average Cd" in df.columns:
        min_cd = df["Average Cd"].min()
        max_cd = df["Average Cd"].max()
    else:
        raise KeyError("CSV に 'Cd' または 'Average Cd' 列が見つかりません。")

    return delta_cd / (max_cd - min_cd)


def inv_scale_cd(
    scaled_cd,
    merge_csv_path="./data/table/all_data.csv",
    noise_data=[]
):
    df = pd.read_csv(merge_csv_path).dropna().reset_index(drop=True)

    if noise_data:
        df = df[~df["folder_name"].isin(noise_data)]

    if "Cd" in df.columns:
        min_cd = df["Cd"].min()
        max_cd = df["Cd"].max()
    elif "Average Cd" in df.columns:
        min_cd = df["Average Cd"].min()
        max_cd = df["Average Cd"].max()
    else:
        raise KeyError("CSV に 'Cd' または 'Average Cd' 列が見つかりません。")

    return scaled_cd * (max_cd - min_cd) + min_cd


# =========================================================
# load mappings & latent codes
# =========================================================

with open(
    "./outputs/data-split/carlatent_new_carfolder2latent_idx.pickle",
    "rb"
) as f:
    carfolder2latent_idx = pickle.load(f)

with open(
    "./outputs/data-split/carlatent_new_latent_idx2carfolder.pickle",
    "rb"
) as f:
    latent_idx2carfolder = pickle.load(f)

latent_codes = torch.load(
    "./outputs/latent-codes/latent_codes_new_array.pth",
    map_location="cpu"
)

num_models = latent_codes.shape[0]


# =========================================================
# select cars
# =========================================================

all_keys = list(carfolder2latent_idx.keys())

if SAMPLE_NUM is None:
    selected_keys = all_keys
else:
    selected_keys = all_keys[:min(SAMPLE_NUM, len(all_keys))]

if ONLY_FIRST_N is not None:
    selected_keys = selected_keys[:ONLY_FIRST_N]

print(f"Selected {len(selected_keys)} car(s) for export.")


# =========================================================
# build models
# =========================================================

def build_deepsdf():
    model = DeepSDF(
        hidden_dim=cfg.hidden_dim,
        xyz_pos_enc_dim=cfg.xyz_pos_enc_dim,
        latent_code_dim=cfg.latent_code_dim,
        dropout_prob=cfg.dropout_prob,
    )

    model.load_state_dict(
        torch.load(
            "./outputs/models/deepsdf_best_epoch4000.pth",
            map_location=cfg.device
        )
    )

    model.to(cfg.device)
    model.eval()
    return model


def build_walker():
    walker = WalkMlpMultiW(
        attribute_dim=1,
        latent_code_dim=cfg.latent_code_dim
    )

    walker.load_state_dict(
        torch.load(
            "./outputs/models/keyword_walker.pth",
            map_location=cfg.device
        )
    )

    walker.to(cfg.device)
    walker.eval()
    return walker


deepsdf = build_deepsdf()
walker = build_walker()


# =========================================================
# load regressor
# =========================================================

regressor = None

if USE_REGRESSOR_FOR_REPORT:
    try:
        reg_path = "./outputs/models/keyword_regressor.pth"

        if os.path.exists(reg_path):
            r = Regressor(
                input_dim=cfg.latent_code_dim,
                output_dim=1
            )

            r.load_state_dict(
                torch.load(
                    reg_path,
                    map_location=cfg.device
                )
            )

            r.to(cfg.device)
            r.eval()
            regressor = r
        else:
            print("Regressor file not found.")

    except Exception as e:
        print("Could not load regressor:", e)
        regressor = None

# =========================================================
# ★ load design predictor models
# =========================================================
try:
    with open('./outputs/models/final_best_gpr_29designs.pkl', 'rb') as f:
        l2d_data = pickle.load(f)
    with open('./outputs/models/pca_pca.pkl', 'rb') as f:
        d2i_data = pickle.load(f)
    print("Successfully loaded design predictor models.")
except Exception as e:
    print("Could not load design predictor models:", e)
    l2d_data = None
    d2i_data = None

# =========================================================
# ★ feature extraction helper
# =========================================================
def get_all_features(latent_tensor):
    """
    潜在表現から、設計変数(29)、意匠性(14)、意匠性主成分(3)を予測して返す関数
    """
    if l2d_data is None or d2i_data is None:
        return {"des": np.zeros(29), "ima": np.zeros(14), "pca": np.zeros(3)}

    latent_np = latent_tensor.detach().cpu().numpy()
    if latent_np.ndim == 1:
        latent_np = latent_np.reshape(1, -1)

    # 1. Latent -> des (29次元)
    latent_scaled = l2d_data['scaler_X'].transform(latent_np)
    preds_norm = l2d_data['model'].predict(latent_scaled)
    
    des_raw = np.zeros((1, 29))
    for i in range(29):
        col = f'des_{i+1}'
        c_min = l2d_data['scaling_params'][col]['min']
        c_max = l2d_data['scaling_params'][col]['max']
        des_raw[0, i] = preds_norm[0, i] * (c_max - c_min) + c_min

    # 2. des -> pca (3主成分)
    # des_15(インデックス14)とdes_22(インデックス21)を除外した27次元を作成
    keep_indices = [i for i in range(29) if i not in [14, 21]]
    preds_27 = des_raw[:, keep_indices]
    preds_27_scaled = d2i_data['scaler_X'].transform(preds_27)
    preds_27_pca = d2i_data['pca_X'].transform(preds_27_scaled)
    
    pca_3 = np.zeros((1, 3))
    for i in range(3):
        model_info = d2i_data['best_models_info'][i]['info']
        if model_info['use_pca']:
            pca_3[0, i] = model_info['model'].predict(preds_27_pca)
        else:
            pca_3[0, i] = model_info['model'].predict(preds_27_scaled).flatten()

    # 3. pca (3主成分) -> ima (14項目) の復元
    ima_scaled = d2i_data['pca_y'].inverse_transform(pca_3)
    ima_raw = d2i_data['scaler_y'].inverse_transform(ima_scaled)

    return {
        "des": des_raw.flatten(),
        "ima": ima_raw.flatten(),
        "pca": pca_3.flatten()
    }


# =========================================================
# prepare step
# =========================================================

delta_step = TARGET_DELTA_CD / max(1, NUM_STEPS)

scaled_step = scale_cd_change(
    delta_step,
    merge_csv_path="./data/table/all_data.csv",
    noise_data=getattr(cfg, "noise_data", [])
)

print(
    f"TARGET_DELTA_CD={TARGET_DELTA_CD}, "
    f"NUM_STEPS={NUM_STEPS}, "
    f"delta_step={delta_step}, "
    f"scaled_step={scaled_step:.7f}, "
    f"EXPORT_BEFORE_FLAG={EXPORT_BEFORE_FLAG}"
)


# =========================================================
# main loop
# =========================================================

errors = []
csv_rows = []

items = list(selected_keys)
pbar = tqdm(items, desc="Processing cars", unit="car")

for idx, carfolder in enumerate(pbar):
    try:
        start_time = time.time()
        pbar.set_description(f"Processing {carfolder}")

        latent_idx = carfolder2latent_idx[carfolder]

        base_latent = (
            latent_codes[latent_idx]
            .unsqueeze(0)
            .to(cfg.device)
            .float()
        )

        # -------------------------------------------------
        # debug
        # -------------------------------------------------

        if PRINT_LATENT_HEAD:
            try:
                tqdm.write(
                    f"[{idx}] {carfolder} "
                    f"latent_idx={latent_idx} "
                    f"norm={base_latent.norm().item():.6e} "
                    f"head={base_latent[0, :6].cpu().numpy().tolist()}"
                )
            except Exception:
                pass

        # -------------------------------------------------
        # BEFORE mesh (optional)
        # -------------------------------------------------

        mesh_export_start = time.time()

        if int(EXPORT_BEFORE_FLAG) == 1:
            before_path = os.path.join(
                OUTPUT_DIR,
                f"{carfolder}_before.ply"
            )

            tqdm.write(
                f"[{carfolder}] Exporting BEFORE mesh -> {before_path}"
            )

            create_mesh(
                model=deepsdf,
                latent=base_latent,
                output_file=before_path,
                N=MESH_RESOLUTION
            )

        # -------------------------------------------------
        # BEFORE Cd & Features
        # -------------------------------------------------

        real_pred_before = np.nan
        scaled_pred_before = np.nan

        # ★ 変形前の各種特徴量を取得
        features_before = get_all_features(base_latent)

        if regressor is not None:
            try:
                with torch.no_grad():
                    scaled_pred_before = (
                        regressor(base_latent.to(cfg.device))
                        .detach()
                        .cpu()
                        .numpy()
                        .ravel()[0]
                    )

                    real_pred_before = inv_scale_cd(
                        scaled_pred_before,
                        merge_csv_path="./data/table/all_data.csv",
                        noise_data=getattr(cfg, "noise_data", [])
                    )

                tqdm.write(
                    f"[{carfolder}] "
                    f"predicted Cd before: "
                    f"scaled={scaled_pred_before:.7f}, "
                    f"real={real_pred_before:.7f}"
                )
            except Exception as e:
                tqdm.write(
                    f"[{carfolder}] "
                    f"regressor before failed: {e}"
                )

        # -------------------------------------------------
        # latent edit
        # -------------------------------------------------

        cur = base_latent.clone()

        for step in range(NUM_STEPS):
            if APPLY_METHOD == "scalar":
                eps = float(scaled_step)

                try:
                    cur = walker(cur, eps)
                except Exception:
                    eps_t = torch.tensor(
                        [[eps]],
                        dtype=torch.float32,
                        device=cfg.device
                    )
                    cur = walker(cur, eps_t)
            else:
                if regressor is None:
                    raise RuntimeError(
                        "delta_via_regressor requires regressor."
                    )

                with torch.no_grad():
                    alpha = regressor(cur).clamp(0.0, 1.0)

                    eps_t = torch.tensor(
                        [[scaled_step]],
                        dtype=torch.float32,
                        device=cfg.device
                    )

                    alpha_prime = (alpha + eps_t).clamp(0.0, 1.0)
                    delta = alpha_prime - alpha

                try:
                    cur = walker(cur, delta)
                except Exception:
                    cur = walker(cur, float(delta.item()))

        edited = cur

        # -------------------------------------------------
        # debug
        # -------------------------------------------------

        try:
            tqdm.write(
                f"[{carfolder}] "
                f"edited latent norm={edited.norm().item():.6e} "
                f"head={edited[0, :6].cpu().numpy().tolist()}"
            )
        except Exception:
            pass

        # -------------------------------------------------
        # AFTER mesh
        # -------------------------------------------------

        after_path = os.path.join(
            OUTPUT_DIR,
            f"{carfolder}_after.ply"
        )

        tqdm.write(
            f"[{carfolder}] Exporting AFTER mesh -> {after_path}"
        )

        create_mesh(
            model=deepsdf,
            latent=edited,
            output_file=after_path,
            N=MESH_RESOLUTION
        )

        mesh_export_end = time.time()
        mesh_export_elapsed = mesh_export_end - mesh_export_start

        if int(EXPORT_BEFORE_FLAG) == 1:
            tqdm.write(
                f"[{carfolder}] mesh export time (before + after): "
                f"{mesh_export_elapsed:.2f} sec"
            )
        else:
            tqdm.write(
                f"[{carfolder}] mesh export time (after only): "
                f"{mesh_export_elapsed:.2f} sec"
            )

        # -------------------------------------------------
        # AFTER Cd & Features
        # -------------------------------------------------

        real_pred_after = np.nan
        scaled_pred_after = np.nan

        # ★ 変形後の各種特徴量を取得
        features_after = get_all_features(edited)

        if regressor is not None:
            try:
                with torch.no_grad():
                    scaled_pred_after = (
                        regressor(edited.to(cfg.device))
                        .detach()
                        .cpu()
                        .numpy()
                        .ravel()[0]
                    )

                    real_pred_after = inv_scale_cd(
                        scaled_pred_after,
                        merge_csv_path="./data/table/all_data.csv",
                        noise_data=getattr(cfg, "noise_data", [])
                    )

                tqdm.write(
                    f"[{carfolder}] "
                    f"predicted Cd after: "
                    f"scaled={scaled_pred_after:.7f}, "
                    f"real={real_pred_after:.7f}"
                )

                delta_scaled = scaled_pred_after - scaled_pred_before
                delta_real = real_pred_after - real_pred_before

                tqdm.write(
                    f"[{carfolder}] "
                    f"predicted ΔCd: "
                    f"scaled={delta_scaled:+.7f}, "
                    f"real={delta_real:+.7f}"
                )

            except Exception as e:
                tqdm.write(
                    f"[{carfolder}] "
                    f"regressor after failed: {e}"
                )

        # -------------------------------------------------
        # CSV row (計算と格納)
        # -------------------------------------------------

        if np.isfinite(real_pred_before) and np.isfinite(real_pred_after):
            after_minus_before_cd = real_pred_after - real_pred_before
        else:
            after_minus_before_cd = np.nan

        # 変化量の計算 (After - Before)
        delta_des = features_after["des"] - features_before["des"]
        delta_ima = features_after["ima"] - features_before["ima"]
        delta_pca = features_after["pca"] - features_before["pca"]

        # 基本情報
        row_dict = {
            "vehicle_id": carfolder,
            "before_cd": real_pred_before,
            "after_cd": real_pred_after,
            "after_minus_before_cd": after_minus_before_cd,
        }
        
        # 動的にカラムを追加
        for i in range(29):
            row_dict[f"delta_des{i+1}"] = delta_des[i]
        for i in range(14):
            row_dict[f"delta_ima{i+1}"] = delta_ima[i]
        for i in range(3):
            row_dict[f"delta_pca{i+1}"] = delta_pca[i]

        csv_rows.append(row_dict)

        # -------------------------------------------------
        # total time
        # -------------------------------------------------

        end_time = time.time()
        elapsed = end_time - start_time

        tqdm.write(
            f"[{carfolder}] "
            f"processing time: {elapsed:.2f} sec"
        )

    except Exception as e:
        errors.append((carfolder, str(e)))
        tqdm.write(f"Error for {carfolder}: {e}")

pbar.close()


# =========================================================
# save CSV
# =========================================================

# 保存するカラムのリストを作成
columns = [
    "vehicle_id",
    "before_cd",
    "after_cd",
    "after_minus_before_cd"
]
columns += [f"delta_des{i+1}" for i in range(29)]
columns += [f"delta_ima{i+1}" for i in range(14)]
columns += [f"delta_pca{i+1}" for i in range(3)]

df_out = pd.DataFrame(
    csv_rows,
    columns=columns
)

df_out.to_csv(OUTPUT_CSV_PATH, index=False)

print("\nFinished.")
print(f"Outputs saved to: {OUTPUT_DIR}")
print(f"Cd & Features CSV saved to: {OUTPUT_CSV_PATH}")

if errors:
    print("\nSome errors occurred:")
    for c, e in errors:
        print(f" - {c}: {e}")
        
