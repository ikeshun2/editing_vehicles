import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# Cd値予測モデル
class CdEstimationModel(nn.Module):
    def __init__(self):
        super(CdEstimationModel, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=1)  # 100x100x100 -> 50x50x50
        self.bn1 = nn.BatchNorm3d(16)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1)  # 50x50x50 -> 25x25x25
        self.bn2 = nn.BatchNorm3d(32)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1)  # 25x25x25 -> 12x12x12
        self.bn3 = nn.BatchNorm3d(64)
        self.conv4 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1)  # 12x12x12 -> 6x6x6
        self.bn4 = nn.BatchNorm3d(128)
        self.conv5 = nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1)  # 6x6x6 -> 3x3x3
        self.bn5 = nn.BatchNorm3d(256)
        self.conv6 = nn.Conv3d(256, 512, kernel_size=3, stride=1, padding=1)  # 3x3x3 -> 3x3x3 (no spatial reduction)
        self.bn6 = nn.BatchNorm3d(512)
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))  # 最終的に (1, 1, 1)
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = F.leaky_relu(self.bn4(self.conv4(x)))
        x = F.leaky_relu(self.bn5(self.conv5(x)))
        x = F.leaky_relu(self.bn6(self.conv6(x)))
        x = self.global_pool(x)
        x = torch.flatten(x, 1)  # Flatten before fully connected layer
        x = self.fc(x)
        return x