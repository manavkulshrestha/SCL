import numpy as np
import torch


class PositionalEncoding(torch.nn.Module):
    def __init__(self, min_deg=0, max_deg=6, scale=0.1,
                 offset=torch.zeros(3)):
        super(PositionalEncoding, self).__init__()
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.n_freqs = max_deg - min_deg + 1
        self.scale = scale
        self.offset = offset

        self.dirs = torch.tensor([
            0.8506508, 0, 0.5257311,
            0.809017, 0.5, 0.309017,
            0.5257311, 0.8506508, 0,
            1, 0, 0,
            0.809017, 0.5, -0.309017,
            0.8506508, 0, -0.5257311,
            0.309017, 0.809017, -0.5,
            0, 0.5257311, -0.8506508,
            0.5, 0.309017, -0.809017,
            0, 1, 0,
            -0.5257311, 0.8506508, 0,
            -0.309017, 0.809017, -0.5,
            0, 0.5257311, 0.8506508,
            -0.309017, 0.809017, 0.5,
            0.309017, 0.809017, 0.5,
            0.5, 0.309017, 0.809017,
            0.5, -0.309017, 0.809017,
            0, 0, 1,
            -0.5, 0.309017, 0.809017,
            -0.809017, 0.5, 0.309017,
            -0.809017, 0.5, -0.309017
        ]).reshape(-1, 3).T

        self.embedding_size = 2 * self.dirs.shape[1] * self.n_freqs + 3

    @staticmethod
    def remove_offset_and_scale(pts: torch.tensor, offset: torch.tensor, scale: torch.tensor) -> torch.tensor:
        if isinstance(offset, torch.Tensor):
            offset = offset.reshape(1, 3)
        if isinstance(scale, torch.Tensor):
            scale = scale.reshape(1, 3)
        pts -= offset
        pts /= scale
        return pts

    def forward(self, tensor: torch.tensor) -> torch.tensor:
        frequency_bands = 2.0 ** torch.linspace(
            self.min_deg, self.max_deg, self.n_freqs,
            dtype=tensor.dtype, device=tensor.device)

        tensor = self.remove_offset_and_scale(tensor, offset=self.offset, scale=self.scale)

        proj = torch.matmul(tensor, self.dirs.to(tensor.device))
        xb = torch.reshape(
            proj[..., None] * frequency_bands,
            list(proj.shape[:-1]) + [-1]
        )
        embedding = torch.sin(torch.cat([xb, xb + 0.5 * np.pi], dim=-1))
        embedding = torch.cat([tensor] + [embedding], dim=-1)

        return embedding
