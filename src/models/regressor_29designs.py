import torch
import torch.nn as nn

class Regressor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        # Batch Normalizationを導入し、適度に深層化することで表現力を確保しつつ安定性を高めます
        # 入力(128) -> 中間層1(128) -> BN -> ReLU -> Dropout
        # 中間層2(64) -> BN -> ReLU -> Dropout -> 出力層(29)
        self.net = nn.Sequential(
            # 第1層: 表現力を維持するために128へ
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2), # 過学習対策
            
            # 第2層: 64へ絞り込み、圧縮的な特徴抽出を行う
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1), # 少し控えめなドロップアウト
            
            # 出力層
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)
        