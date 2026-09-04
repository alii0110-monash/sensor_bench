"""Sensor token extraction (frozen AlignmentModel encoders) + LLM-space projector."""
from __future__ import annotations
import torch
import torch.nn as nn

from framework.models.alignment import AlignmentModel, MODALITIES


def load_frozen_encoders(ckpt_path: str, device: str) -> AlignmentModel:
    """Instantiate AlignmentModel and load only the encoder weights from an
    m6b-style checkpoint (projection_head/classification_head ignored)."""
    model = AlignmentModel(text_dim=512)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    enc_sd = {k: v for k, v in sd.items() if k.startswith("encoders.")}
    missing, unexpected = model.load_state_dict(enc_sd, strict=False)
    enc_missing = [k for k in missing if k.startswith("encoders.")]
    assert not enc_missing, f"encoder weights missing from checkpoint: {enc_missing[:6]}"
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def extract_tokens(model: AlignmentModel, mods: dict) -> torch.Tensor:
    """Full-profile token extraction: (B, 5, N_TOK, D=256)."""
    avail = {m: True for m in MODALITIES}
    return model.encode_modalities(mods, avail)


class SensorTokenProjector(nn.Module):
    """(B, M, N, 256) -> (B, M*N, hidden): per-token linear + modality type
    embedding + LayerNorm (stable across量纲 differences)."""

    def __init__(self, in_dim: int = 256, hidden: int = 896, num_modalities: int = 5):
        super().__init__()
        self.num_modalities = num_modalities
        self.linear = nn.Linear(in_dim, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.type_emb = nn.Parameter(torch.randn(num_modalities, hidden) * 0.02)

    def forward(self, toks: torch.Tensor) -> torch.Tensor:
        B, M, N, D = toks.shape
        x = self.linear(toks.reshape(B, M * N, D))
        x = self.norm(x)
        type_seq = self.type_emb.unsqueeze(1).expand(M, N, -1).reshape(M * N, -1)
        return x + type_seq.unsqueeze(0)
