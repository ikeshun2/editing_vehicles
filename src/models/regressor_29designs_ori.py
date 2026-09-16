import torch
import torch.nn as nn


class Regressor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        # 【修正ポイント1】層を浅くし、ニューロン数を削減 (128 -> 64 -> output)
        # 複雑すぎるモデルによる「訓練データの丸暗記（過学習）」を防ぎます
        self.fc1 = nn.utils.weight_norm(nn.Linear(input_dim, 64))
        self.fc2 = nn.utils.weight_norm(nn.Linear(64, output_dim))
        
        # 【修正ポイント2】Dropoutの調整
        # ネットワークが浅くなった分、丸暗記を防ぐブレーキを少し強め(5% -> 10%)に設定
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 入力(128) -> 隠れ層(64) -> 活性化 -> Dropout -> 出力層(29)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
        