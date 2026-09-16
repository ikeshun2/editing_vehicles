import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

# 自プロジェクトパスを追加
sys.path.append("./")

from experimet_config import Config
from src.utils.utils import set_seed
from src.datasets.latent_navigation_dataset import KeywordRegressorDataset
from src.models.regressor import Regressor

# =========================================================
# 設定・準備
# =========================================================
set_seed(42)
cfg = Config()

# デバッグフラグ
DEBUG_SHAPES = True

# 出力ディレクトリ
os.makedirs("./outputs/models", exist_ok=True)

# =========================================================
# keyword columns
# =========================================================
keyword_columns = cfg.keyword_attribute

# =========================================================
# latent index 辞書読み込み
# =========================================================
with open("./outputs/data-split/carfolder2latent_idx.pickle", "rb") as f:
    carfolder2latent_idx = pickle.load(f)

with open("./outputs/data-split/latent_idx2carfolder.pickle", "rb") as f:
    latent_idx2carfolder = pickle.load(f)

# =========================================================
# latent codes 読み込み
# =========================================================
latent_codes = torch.load("./outputs/latent-codes/latent_codes_best_epoch4000.pth")

print("latent_codes shape:", latent_codes.shape)
print("num latent vehicles:", len(carfolder2latent_idx))

# =========================================================
# merged.csv 読み込み
# =========================================================
merge_df = pd.read_csv("./data/table/merged.csv")
merge_df = merge_df.dropna().reset_index(drop=True)

print("original merge_df size:", len(merge_df))

# =========================================================
# DeepSDF latent が存在する車両のみ残す
# =========================================================
valid_carfolders = set(carfolder2latent_idx.keys())

# merge_df 内の carfolder が latent に存在するものだけ残す
merge_df = merge_df[
    merge_df["carfolder"].isin(valid_carfolders)
].reset_index(drop=True)

print("filtered merge_df size:", len(merge_df))

# =========================================================
# missing key 確認
# =========================================================
csv_keys = set(merge_df["carfolder"].unique())
missing_keys = csv_keys - valid_carfolders

print("missing keys:", missing_keys)
print("num missing:", len(missing_keys))

# =========================================================
# scaling (0~1)
# =========================================================
for col in keyword_columns:
    merge_df[col] = (
        merge_df[col] - merge_df[col].min()
    ) / (
        merge_df[col].max() - merge_df[col].min()
    )

# =========================================================
# train / valid split
# =========================================================
train_df, valid_df = train_test_split(
    merge_df,
    test_size=0.2,
    random_state=42,
    shuffle=True,
)

print("train size:", len(train_df))
print("valid size:", len(valid_df))

# =========================================================
# Dataset / DataLoader
# =========================================================
train_dataset = KeywordRegressorDataset(
    keyword_columns,
    train_df,
    latent_codes,
    carfolder2latent_idx,
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=cfg.batch_size,
    shuffle=True,
    drop_last=True,
)

valid_dataset = KeywordRegressorDataset(
    keyword_columns,
    valid_df,
    latent_codes,
    carfolder2latent_idx,
)

valid_dataloader = DataLoader(
    valid_dataset,
    batch_size=cfg.batch_size,
    shuffle=False,
    drop_last=False,
)

print("num train batches:", len(train_dataloader))
print("num valid batches:", len(valid_dataloader))

# =========================================================
# model
# =========================================================
regressor = Regressor(
    input_dim=cfg.latent_code_dim,
    output_dim=len(keyword_columns),
).to(cfg.device)

# =========================================================
# optimizer
# =========================================================
optimizer = torch.optim.Adam(
    [
        {
            "params": regressor.parameters(),
            "lr": 1e-3,
        },
    ]
)

# =========================================================
# scheduler
# =========================================================
num_train_batches = max(1, len(train_dataloader))
num_train_steps = max(
    1,
    cfg.keyword_reg_epoch * num_train_batches
)

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=max(0, cfg.keyword_reg_warmup_epoch),
    num_training_steps=num_train_steps,
)

# =========================================================
# loss
# =========================================================
criterion = nn.MSELoss()

# =========================================================
# helper
# =========================================================
def fix_pred_and_target_shapes(pred, target, output_dim):

    # unpack if tuple/list
    if isinstance(pred, (list, tuple)):
        pred = pred[0]

    pred = torch.as_tensor(pred).float()
    target = torch.as_tensor(target).float()

    # squeeze unnecessary singleton dims
    while pred.dim() > 2 and pred.size(-1) == 1:
        pred = pred.squeeze(-1)

    while target.dim() > 2 and target.size(-1) == 1:
        target = target.squeeze(-1)

    # latent_code が [B,1,128] の場合 -> [B,128]
    if pred.dim() == 3 and pred.size(1) == 1:
        pred = pred.squeeze(1)

    if target.dim() == 3 and target.size(1) == 1:
        target = target.squeeze(1)

    batch_size = pred.size(0)

    # pred reshape
    try:
        pred = pred.view(batch_size, output_dim)
    except RuntimeError:
        pred = pred.reshape(batch_size, -1)

        if pred.size(1) >= output_dim:
            pred = pred[:, :output_dim]
        else:
            pred = torch.nn.functional.pad(
                pred,
                (0, output_dim - pred.size(1))
            )

    # target reshape
    try:
        if target.dim() == 1 and output_dim == 1:
            target = target.view(-1, 1)
        else:
            target = target.view(batch_size, output_dim)

    except RuntimeError:
        target = target.reshape(batch_size, -1)

        if target.size(1) >= output_dim:
            target = target[:, :output_dim]
        else:
            target = torch.nn.functional.pad(
                target,
                (0, output_dim - target.size(1))
            )

    return pred, target

# =========================================================
# training
# =========================================================
best_loss = np.inf
best_model = None
output_dim = len(keyword_columns)

for epoch in range(cfg.keyword_reg_epoch):

    # =====================================================
    # train
    # =====================================================
    regressor.train()

    train_loss_sum = 0.0
    train_samples = 0

    for batch_idx, batch in enumerate(train_dataloader):

        latent_code = batch["latent_code"].to(cfg.device).float()
        raw_keyword = batch["keyword"]

        # latent_code が [B,1,128] -> [B,128]
        if latent_code.dim() == 3 and latent_code.size(1) == 1:
            latent_code = latent_code.squeeze(1)

        # debug
        if DEBUG_SHAPES and epoch == 0 and batch_idx == 0:
            print(
                "DEBUG TRAIN latent_code shape:",
                latent_code.shape
            )

            print(
                "DEBUG TRAIN keyword shape:",
                torch.as_tensor(raw_keyword).shape
            )

        # forward
        pred_keyword = regressor(latent_code)

        # shape fix
        pred_keyword, keyword = fix_pred_and_target_shapes(
            pred_keyword,
            raw_keyword.to(cfg.device),
            output_dim,
        )

        optimizer.zero_grad()

        loss = criterion(pred_keyword, keyword)

        loss.backward()

        optimizer.step()

        scheduler.step()

        batch_size = latent_code.size(0)

        train_loss_sum += loss.item() * batch_size
        train_samples += batch_size

    # =====================================================
    # validation
    # =====================================================
    regressor.eval()

    valid_loss_sum = 0.0
    valid_samples = 0

    with torch.no_grad():

        for vbatch_idx, batch in enumerate(valid_dataloader):

            latent_code = batch["latent_code"].to(cfg.device).float()
            raw_keyword = batch["keyword"]

            # latent_code が [B,1,128] -> [B,128]
            if latent_code.dim() == 3 and latent_code.size(1) == 1:
                latent_code = latent_code.squeeze(1)

            # debug
            if DEBUG_SHAPES and epoch == 0 and vbatch_idx == 0:
                print(
                    "DEBUG VALID latent_code shape:",
                    latent_code.shape
                )

                print(
                    "DEBUG VALID keyword shape:",
                    torch.as_tensor(raw_keyword).shape
                )

            # forward
            pred_keyword = regressor(latent_code)

            # shape fix
            pred_keyword, keyword = fix_pred_and_target_shapes(
                pred_keyword,
                raw_keyword.to(cfg.device),
                output_dim,
            )

            loss = criterion(pred_keyword, keyword)

            batch_size = latent_code.size(0)

            valid_loss_sum += loss.item() * batch_size
            valid_samples += batch_size

    # =====================================================
    # average loss
    # =====================================================
    train_avg = train_loss_sum / max(1, train_samples)
    valid_avg = valid_loss_sum / max(1, valid_samples)

    print(
        f"epoch: {epoch+1}/{cfg.keyword_reg_epoch}, "
        f"train_loss: {train_avg:.6f}, "
        f"valid_loss: {valid_avg:.6f}"
    )

    # =====================================================
    # save best
    # =====================================================
    if valid_avg < best_loss:

        best_loss = valid_avg

        best_model = regressor.state_dict()

        torch.save(
            best_model,
            "./outputs/models/keyword_regressor.pth"
        )

        print(
            f"  -> saved best model "
            f"(valid_loss: {best_loss:.6f})"
        )

print(f"Finished. best_valid_loss: {best_loss:.6f}")
