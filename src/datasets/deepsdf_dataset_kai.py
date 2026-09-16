import os
import torch
import numpy as np
from torch.utils.data import Dataset


def unpack_sdf_samples(points, sdf, subsample=50000, surface_threshold=0.01):
    points = torch.tensor(points).reshape(-1, 3)
    sdf = torch.tensor(sdf).reshape(-1, 1)
    samples = torch.cat([points, sdf], -1).reshape(-1, 4)

    # 表面付近の点を抽出
    near_surface = torch.abs(samples[:, 3]) < surface_threshold
    near_surface_samples = samples[near_surface]

    # 表面付近以外の点を抽出
    far_surface_samples = samples[~near_surface]

    # 表面付近を優先的にサンプリング
    num_near_samples = int(subsample * 0.75)  # 75%を表面付近からサンプリング
    num_far_samples = subsample - num_near_samples  # 残りを遠い点からサンプリング

    near_idx = np.arange(0, len(near_surface_samples))
    far_idx = np.arange(0, len(far_surface_samples))
    np.random.shuffle(near_idx)
    np.random.shuffle(far_idx)

    sample_near = near_surface_samples[near_idx[:num_near_samples], :]
    sample_far = far_surface_samples[far_idx[:num_far_samples], :]

    samples = torch.cat([sample_near, sample_far], 0)

    xyz = samples[:, :3]
    sdf = samples[:, 3].reshape(-1, 1)

    return xyz, sdf



class DeepSDFDataset(Dataset):
    def __init__(self, points_path, carfolder2latent_idx, subsample=50000):
        super().__init__()
        self.subsample = subsample
        self.carfolder2latent_idx = carfolder2latent_idx

        self.idx2training_data = {}

        # read points.npy and sdf.npy
        for i, path in enumerate(points_path):
            assert os.path.exists(path), f"{path} does not exist."

            car_foldername = path.split("/")[-2]
            points_path = path
            sdf_path = path.replace("points.npy", "sdf.npy")

            points = np.load(points_path)
            sdf = np.load(sdf_path)

            self.idx2training_data[i] = {
                "car_foldername": car_foldername,
                "points": points,
                "sdf": sdf,
            }

    def __len__(self):
        return len(self.idx2training_data)

    def __getitem__(self, idx):
        car_foldername = self.idx2training_data[idx]["car_foldername"]
        points = self.idx2training_data[idx]["points"]
        sdf = self.idx2training_data[idx]["sdf"]

        points, sdf = unpack_sdf_samples(points, sdf, subsample=self.subsample)

        return {
            "car_foldername": car_foldername,
            "points": points,
            "sdf": sdf,
        }
