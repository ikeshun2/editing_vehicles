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

sys.path.append("./")
from experimet_config import Config
from src.utils.utils import set_seed
from src.datasets.latent_navigation_dataset import KeywordNavigationDataset
from src.models.regressor import Regressor
from src.models.latent_walker import WalkMlpMultiW
from src.utils.metric import regressor_criterion

# ==================================================
# PyTorch版の意匠性代理モデルクラス定義
# ==================================================
class DesignEvaluator(nn.Module):
    def __init__(self, latent_dim=128, output_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, x):
        return self.net(x)

set_seed(42)
cfg = Config()

# loading keyword columns
keyword_columns = cfg.keyword_attribute

# loading training data
merge_df = pd.read_csv("./data/table/merged.csv").dropna().reset_index(drop=True)
merge_df = merge_df[~merge_df["folder_name"].isin(cfg.noise_data)]

for col in keyword_columns:
    # 0~1にmin-max scaling
    merge_df[col] = (merge_df[col] - merge_df[col].min()) / (
        merge_df[col].max() - merge_df[col].min()
    )

# loading carfolderlatent_idx, latent_idx2carfolder
with open("./outputs/data-split/carfolder2latent_idx.pickle", "rb") as f:
    carfolder2latent_idx = pickle.load(f)

with open("./outputs/data-split/latent_idx2carfolder.pickle", "rb") as f:
    latent_idx2carfolder = pickle.load(f)

# loading latent codes
latent_codes = torch.load("./outputs/latent-codes/latent_codes_best_epoch4000.pth")

# loading regressor model
regressor = (
    Regressor(
        input_dim=cfg.latent_code_dim,
        output_dim=len(keyword_columns),
    )
    .to(cfg.device)
    .eval()
)

regressor.load_state_dict(
    torch.load("./outputs/models/keyword_regressor.pth")
)

# --------------------------------------------------
# 意匠性予測用のPyTorch代理モデルをロード・凍結
# --------------------------------------------------
design_evaluator = DesignEvaluator(latent_dim=cfg.latent_code_dim, output_dim=3).to(cfg.device)
design_evaluator.load_state_dict(torch.load("./outputs/models/design_surrogate.pth"))
design_evaluator.eval()

# Freeze parameters
for p in regressor.parameters():
    p.requires_grad = False
for p in design_evaluator.parameters():
    p.requires_grad = False

# create latent walker model
latent_walker = WalkMlpMultiW(
    attribute_dim=len(keyword_columns),
    latent_code_dim=cfg.latent_code_dim,
).to(cfg.device)

# create dataset
train_dataset = KeywordNavigationDataset(
    keyword_columns,
    cfg.latent_code_dim,
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=cfg.keyword_walker_batch_size,
    shuffle=True,
    drop_last=True,
)

# create optimizer
reg_criterion = regressor_criterion
latent_criterion = nn.MSELoss() # ★追加: 潜在表現の逸脱を防ぐアンカー用の損失関数

optimizer = torch.optim.Adam(
    [
        {
            "params": latent_walker.parameters(),
            "lr": cfg.keyword_walker_initial_lr,
        },
    ]
)

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=cfg.keyword_walker_warmup_epoch,
    num_training_steps=cfg.keyword_walker_epoch * len(train_dataloader),
)

# training
best_loss = np.inf
best_model = None

# マージン設定：提供されたサンプルのスケールを考慮
# margin = torch.tensor([0.2, 0.3, 0.2], device=cfg.device)
margin = torch.tensor([0, 0, 0], device=cfg.device)

# CSVの出力ヘッダも更新
csv_columns = ["epoch", "L_reg", "L_design", "L_latent", "total_loss"]
csv_data = []

for epoch in range(cfg.keyword_walker_epoch):

    total_loss = 0.0
    total_reg_loss = 0.0
    total_design_loss = 0.0
    total_latent_loss = 0.0

    latent_walker.train()

    for idx, batch in enumerate(train_dataloader):

        random_latent_code = batch["random_latent_code"].to(cfg.device)
        epsilon = batch["epsilon"].to(cfg.device)

        optimizer.zero_grad()

        # [Loss項1] 目的の編集量に対する最適化 (Cd値の低減)
        alpha = regressor(random_latent_code)
        delta = torch.clip(alpha + epsilon, 0, 1) - alpha
        z_prime = latent_walker(random_latent_code, delta)
        
        alpha_prime = alpha + delta
        alpha_hat_prime = regressor(z_prime)
        reg_loss = reg_criterion(alpha_hat_prime, alpha_prime)

        # [Loss項2] 意匠性のマージン付き保持
        design_pre = design_evaluator(random_latent_code)
        design_post = design_evaluator(z_prime)
        
        diff = torch.abs(design_post - design_pre)
        design_loss = torch.mean(torch.relu(diff - margin))

        # [Loss項3] ★新規追加: 潜在空間からの逸脱防止（アンカー）
        # これがないと、AIがズルをしてz_primeを無限の彼方に飛ばしてしまいます
        latent_loss = latent_criterion(z_prime, random_latent_code)

        # --- 重みの設定 ---
        LAMBDA_REG = 1       # 例: 1.0
        LAMBDA_DESIGN = 10   # 例: 10.0
        LAMBDA_LATENT = 10                                # 発散防止の重み (暴走するなら数値を上げる)
        
        loss = (
            LAMBDA_REG * reg_loss
            + LAMBDA_DESIGN * design_loss
            + LAMBDA_LATENT * latent_loss
        )

        loss.backward()

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_reg_loss += reg_loss.item()
        total_design_loss += design_loss.item()
        total_latent_loss += latent_loss.item()

    avg_loss = total_loss / len(train_dataloader)
    avg_reg_loss = total_reg_loss / len(train_dataloader)
    avg_design_loss = total_design_loss / len(train_dataloader)
    avg_latent_loss = total_latent_loss / len(train_dataloader)

    print(
        f"epoch: {epoch:03d}, "
        f"loss: {avg_loss:.4f}, "
        f"L_reg: {avg_reg_loss:.4f}, "
        f"L_design: {avg_design_loss:.4f}, "
        f"L_latent: {avg_latent_loss:.4f}"
    )

    csv_data.append([epoch, avg_reg_loss, avg_design_loss, avg_latent_loss, avg_loss])

    if best_loss > avg_loss:
        best_loss = avg_loss
        best_model = latent_walker.state_dict()

    torch.save(best_model, "./outputs/models/keyword_walker.pth")

df = pd.DataFrame(csv_data, columns=csv_columns)
df.to_csv("./outputs/models/keyword_walker_losses.csv", index=False)

