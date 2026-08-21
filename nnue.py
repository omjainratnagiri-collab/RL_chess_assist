import torch
import torch.nn as nn
import sys
from pathlib import Path

from feature_transformer import FeatureTransformer, ClippedReLU
from halfkp_encoding import NUM_FEATURES

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = ClippedReLU()

    def forward(self, x):
        residual = x
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return self.act(x + residual)


class NNUE(nn.Module):

    def __init__(
        self,
        num_features=NUM_FEATURES,
        embedding_dim=160,                                                         
                                                                               
                                                                              
                                                                         
                                                                              
                                                                       
        hidden_dim=256,
        policy_size=4672,
    ):
        super().__init__()

        self.feature_transformer = FeatureTransformer(
            num_features=num_features,
            embedding_dim=embedding_dim,
        )

        self.input_proj = nn.Linear(embedding_dim * 2, hidden_dim)
        self.act = ClippedReLU()

        self.shared = ResidualBlock(hidden_dim)

        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, policy_size),
        )

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, us_batch, them_batch):
        x = self.feature_transformer(us_batch, them_batch)
        x = self._trunk_and_heads(x)
        return x

    def forward_from_accumulators(self, us_vec, them_vec, device=None):
        
        dev = device if device is not None else next(self.parameters()).device
        us_t = torch.as_tensor(us_vec, dtype=torch.float32, device=dev).unsqueeze(0)
        them_t = torch.as_tensor(them_vec, dtype=torch.float32, device=dev).unsqueeze(0)
        x = torch.cat([self.feature_transformer.activation(us_t),
                        self.feature_transformer.activation(them_t)], dim=1)
        return self._trunk_and_heads(x)

    def _trunk_and_heads(self, x):
        x = self.act(self.input_proj(x))
        x = self.shared(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value
