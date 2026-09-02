from __future__ import annotations
import torch
import torch.nn as nn

D = 256
N_TOK = 16


class WifiEncoder(nn.Module):
    """(B,5,3,114,10) -> (B,16,D); temporal=True returns (B,T,16,D)."""

    def __init__(self, temporal: bool = False):
        super().__init__()
        self.temporal = temporal
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU())
        self.proj = nn.Linear(64, D)

    def forward(self, x):
        B, T = x.shape[:2]
        x = x.reshape(B * T, *x.shape[2:])
        x = self.conv(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (4, 4))
        x = x.flatten(2).transpose(1, 2)
        out = self.proj(x).view(B, T, N_TOK, -1)
        return out if self.temporal else out.mean(dim=1)


class DepthEncoder(nn.Module):
    """(B,5,1,224,224) -> (B,16,D); temporal=True returns (B,T,16,D)."""

    def __init__(self, temporal: bool = False):
        super().__init__()
        self.temporal = temporal
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU())
        self.proj = nn.Linear(64, D)

    def forward(self, x):
        B, T = x.shape[:2]
        x = x.reshape(B * T, *x.shape[2:])
        x = self.conv(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (4, 4))
        x = x.flatten(2).transpose(1, 2)
        out = self.proj(x).view(B, T, N_TOK, -1)
        return out if self.temporal else out.mean(dim=1)


class PointEncoder(nn.Module):
    """(B,5,P,C) -> (B,16,D); temporal=True returns (B,T,16,D). lidar: C=3, mmwave: C=5."""

    def __init__(self, in_c: int, temporal: bool = False):
        super().__init__()
        self.temporal = temporal
        self.mlp = nn.Sequential(
            nn.Conv1d(in_c, 64, 1), nn.ReLU(),
            nn.Conv1d(64, 64, 1), nn.ReLU())
        self.proj = nn.Linear(64, D)

    def forward(self, x):
        B, T = x.shape[:2]
        x = x.reshape(B * T, *x.shape[2:]).transpose(1, 2)  # (BT,C,P)
        x = self.mlp(x)
        x = torch.nn.functional.adaptive_avg_pool1d(x, N_TOK)  # (BT,C,16)
        out = self.proj(x.transpose(1, 2)).view(B, T, N_TOK, -1)
        return out if self.temporal else out.mean(dim=1)


class MLPEncoder(nn.Module):
    """Encode a 1-D structured-feature vector (e.g. v5_structfeat mmwave 134d,
    wifi 161d, depth 63d, lidar 353d) -> (B,16,D).

    Used to make the main pipeline (token_fusion / late_fusion) work with the
    domain-aware structured features produced by
    framework.eval.dataset_quality.feature_extract. Input has no time axis:
    (B, F) where F is the feature dimension.
    """

    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, N_TOK * D))
        self.proj = nn.Linear(N_TOK * D, N_TOK * D)

    def forward(self, x):
        # x: (B, L) -> (B, 16, D)
        B = x.shape[0]
        h = self.mlp(x)  # (B, 16*D)
        return self.proj(h).view(B, N_TOK, D)
