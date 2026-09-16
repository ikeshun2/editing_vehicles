import os
import torch
import numpy as np
from torch.utils.data import Dataset


class KeywordRegressorDataset(Dataset):
    def __init__(self, keyword_columns, merge_df, latent_codes, carfolder2latent_idx):
        super().__init__()
        self.merge_df = merge_df.dropna().reset_index(drop=True)
        self.latent_codes = latent_codes
        self.carfolder2latent_idx = carfolder2latent_idx
        self.carfolders = list(carfolder2latent_idx.keys())
        self.keyword_columns = keyword_columns
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 存在しない carfolder を記録するリスト
        self.missing_carfolders = []

    def __len__(self):
        return len(self.merge_df)

    def __getitem__(self, idx):
        row = self.merge_df.loc[idx]
        carfolder = row["folder_name"]
        
        # carfolder が carfolder2latent_idx に存在するか確認
        if carfolder not in self.carfolder2latent_idx:
            print(f"Warning: The key '{carfolder}' is missing in carfolder2latent_idx.")
            self.missing_carfolders.append(carfolder)  # 存在しない場合はリストに追加
            return None  # 存在しない場合は None を返す
        
        latent_idx = self.carfolder2latent_idx[carfolder]
        latent_code = self.latent_codes[latent_idx].unsqueeze(0).to(self.device)
        keyword = row[self.keyword_columns].values.astype(np.float32)

        return {
            "latent_code": latent_code,
            "keyword": keyword,
        }

    def print_missing_carfolders(self):
        if self.missing_carfolders:
            print("The following carfolders are missing in carfolder2latent_idx:")
            for carfolder in self.missing_carfolders:
                print(carfolder)
        else:
            print("No missing carfolders.")


class KeywordNavigationDataset(Dataset):
    def __init__(self, keyword_columns, latent_code_dim):
        super().__init__()
        self.keyword_columns = keyword_columns
        self.latent_code_dim = latent_code_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        random_noise_latent_code = torch.normal(0, 1e-1, size=(1, self.latent_code_dim))
        random_noise_latent_code = random_noise_latent_code.squeeze(0).to(self.device)

        epsilon = 2 * torch.rand(1, len(self.keyword_columns)) - 1  # 一様分布からサンプリング

        epsilon = epsilon.squeeze(0).to(self.device)

        return {
            "random_latent_code": random_noise_latent_code,
            "epsilon": epsilon,
        }


class GeometryRegressorDataset(Dataset):
    def __init__(
        self, geometry_columns, geometry_df, latent_codes, carfolder2latent_idx
    ):
        super().__init__()
        self.geometry_df = geometry_df.dropna().reset_index(drop=True)
        self.latent_codes = latent_codes
        self.carfolder2latent_idx = carfolder2latent_idx
        self.carfolders = list(carfolder2latent_idx.keys())
        self.geometry_columns = geometry_columns
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 存在しない carfolder を記録するリスト
        self.missing_carfolders = []

    def __len__(self):
        return len(self.geometry_df)

    def __getitem__(self, index):
        row = self.geometry_df.loc[index]
        carfolder = row["folder_name"]
        
        # carfolder が carfolder2latent_idx に存在するか確認
        if carfolder not in self.carfolder2latent_idx:
            print(f"Warning: The key '{carfolder}' is missing in carfolder2latent_idx.")
            self.missing_carfolders.append(carfolder)  # 存在しない場合はリストに追加
            return None  # 存在しない場合は None を返す
        
        latent_idx = self.carfolder2latent_idx[carfolder]
        latent_code = self.latent_codes[latent_idx].unsqueeze(0).to(self.device)
        geometry = row[self.geometry_columns].values.astype(np.float32)

        return {
            "latent_code": latent_code,
            "geometry": geometry,
        }

    def print_missing_carfolders(self):
        if self.missing_carfolders:
            print("The following carfolders are missing in carfolder2latent_idx:")
            for carfolder in self.missing_carfolders:
                print(carfolder)
        else:
            print("No missing carfolders.")


class GeometryNavigationDataset(Dataset):
    def __init__(self, geometry_columns, latent_code_dim):
        super().__init__()
        self.geometry_columns = geometry_columns
        self.latent_code_dim = latent_code_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        random_noise_latent_code = torch.normal(0, 1e-1, size=(1, self.latent_code_dim))
        random_noise_latent_code = random_noise_latent_code.squeeze(0).to(self.device)

        epsilon = 2 * torch.rand(1, len(self.geometry_columns)) - 1  # 一様分布からサンプリング
        epsilon = epsilon.squeeze(0).to(self.device)

        return {
            "random_latent_code": random_noise_latent_code,
            "epsilon": epsilon,
        }
