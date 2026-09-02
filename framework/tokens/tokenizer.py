"""CanonicalTokenizer: 传感器样本 ↔ CanonicalToken (spec M6a 组件 2).

冻结 AlignmentModel + PerceiverProjection, 传感器 → 规范空间 token."""
from __future__ import annotations

import torch
import numpy as np

from ..models.alignment import AlignmentModel
from ..models.perceiver import PerceiverProjection
from .canonical import CanonicalToken

MODALITY_ORDER = ["wifi", "depth", "lidar", "mmwave", "rgb"]


class CanonicalTokenizer:
    def __init__(self, align_ckpt: str, proj_ckpt: str, k: int = 8, device: str = "cpu"):
        self.k = k
        self.device = device if device == "cuda" and torch.cuda.is_available() else "cpu"
        self.align = AlignmentModel(num_modalities=5, text_dim=512)
        self.align.projection_head = torch.nn.Sequential(
            torch.nn.Linear(256, 27), torch.nn.Linear(27, 512))   # 原型头
        self.align.load_state_dict(torch.load(align_ckpt, map_location="cpu"), strict=False)
        self.align.eval().to(self.device)
        for p in self.align.parameters():
            p.requires_grad_(False)
        self.proj = PerceiverProjection(out_dim=4096, k=k).to(self.device)
        self.proj.load_state_dict(torch.load(proj_ckpt, map_location="cpu"))
        self.proj.eval()
        for p in self.proj.parameters():
            p.requires_grad_(False)

    def encode(self, sample) -> CanonicalToken:
        """传感器样本 → CanonicalToken."""
        mods = {m: torch.from_numpy(sample.modalities[m].data)[None].to(self.device)
                for m in MODALITY_ORDER if m in sample.modalities}
        avail = {m: True for m in mods}
        with torch.no_grad():
            ct = self.align.encode_modalities(mods, avail)   # (1, M, 16, 256)
            pe = self.proj(ct)                               # (1, M*k, 4096)
        data = pe[0].cpu().numpy().astype(np.float32)
        return CanonicalToken(id=sample.id, label=sample.label, data=data,
                              modality_order=MODALITY_ORDER, k=self.k,
                              meta={"encoder_version": "alignment_seed0+proj_verb"})

    def decode(self, tok: CanonicalToken) -> torch.Tensor:
        """CanonicalToken → (1, M*k, 4096) 张量.
        注: 返回带 batch 维 (1, M*k, H) 以便与 LLMAdapter.inject 拼接一致;
        spec 签名写 np.ndarray (M*k,4096) 为简化, 此处以张量实现为准."""
        tok.validate()
        return torch.from_numpy(tok.data)[None].to(self.device)
