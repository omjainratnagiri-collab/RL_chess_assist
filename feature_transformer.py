import torch
import torch.nn as nn


class ClippedReLU(nn.Module):
    """clamp(x, 0, 1) -- what real NNUE implementations use, more stable
    than unbounded ReLU on top of a summed EmbeddingBag feature vector."""

    def forward(self, x):
        return torch.clamp(x, 0.0, 1.0)


class FeatureTransformer(nn.Module):
    

    def __init__(self, num_features, embedding_dim):
        super().__init__()
        self.embedding = nn.EmbeddingBag(num_features, embedding_dim, mode="sum")
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        self.activation = ClippedReLU()

    def _embed(self, indices_batch):
        device = self.embedding.weight.device
        flat, offsets, total = [], [0], 0
        for indices in indices_batch:
            flat.extend(indices)
            total += len(indices)
            offsets.append(total)
        flat = torch.tensor(flat, dtype=torch.long, device=device)
        offsets = torch.tensor(offsets[:-1], dtype=torch.long, device=device)
        return self.activation(self.embedding(flat, offsets))

    def forward(self, us_batch, them_batch):
        us = self._embed(us_batch)
        them = self._embed(them_batch)
        return torch.cat([us, them], dim=1)
