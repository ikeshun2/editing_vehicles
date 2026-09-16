import torch
import torch.nn as nn


class WalkMlpMultiW(nn.Module):
    def __init__(self, attribute_dim, latent_code_dim=128):
        super(WalkMlpMultiW, self).__init__()

        self.linear = nn.Sequential(
            *[
                nn.Linear(latent_code_dim, 2 * latent_code_dim),
                nn.LeakyReLU(0.2, True),
                nn.Linear(2 * latent_code_dim, 2 * latent_code_dim),
                nn.LeakyReLU(0.2, True),
                nn.Linear(2 * latent_code_dim, attribute_dim),
            ]
        )

        self.embed = nn.Linear(attribute_dim, latent_code_dim, bias=False)

    def forward(self, latent_codes, delta, lambda_=3):
        # out [batch_size, attribute_dim]
        out = self.linear(latent_codes)
        # out after normalization ( 归一化) 后乘以变化量
        out = out / torch.norm(out, dim=1, keepdim=True) * lambda_
        # 
        updated_latent_codes = latent_codes + self.embed(delta * out)

        return updated_latent_codes
