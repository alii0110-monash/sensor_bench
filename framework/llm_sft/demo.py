"""Interactive demo engine: sensor sample -> (optional modality toggles) ->
LLM answer. Thin, injectable core so Streamlit stays dumb and tests stay tiny."""
from __future__ import annotations
import os

import torch

from framework.llm_sft.classmap import load_class_map, match_scores
from framework.llm_sft.dataset import collate_mods, load_split_base
from framework.llm_sft.projector import extract_tokens
from framework.llm_sft.prompting import (QUESTION, batched_prompt_embeds,
                                         greedy_decode)
from framework.models.alignment import MODALITIES

DEFAULT_QUESTION = QUESTION


def collate_mods_avail(samples, avail, device):
    """Stack only available modalities; collate_mods requires all 5 — this
    variant feeds encode_modalities' zero-slot path for missing ones."""
    import numpy as np
    mods = {}
    for m in MODALITIES:
        if avail.get(m, True):
            mods[m] = torch.from_numpy(np.stack(
                [s.modalities[m].data for s in samples]).astype("float32")).to(device)
    return mods


class DemoEngine:
    def __init__(self, model, proj, alignment, tok, samples, class_map, device="cpu"):
        self.model, self.proj, self.alignment, self.tok = model, proj, alignment, tok
        self.samples = samples
        self.class_map = class_map
        self.device = device
        self.pre_ids, self.post_ids = None, None
        self._question = None

    def _prompt(self, question):
        if self.pre_ids is None or question != self._question:
            from framework.llm_sft.prompting import encode_prompt_ids
            self.pre_ids, self.post_ids = encode_prompt_ids(self.tok, question)
            self._question = question
        return self.pre_ids, self.post_ids

    @torch.no_grad()
    def answer(self, sample, avail: dict | None = None,
               question: str | None = None, max_new_tokens: int = 16) -> dict:
        avail = avail or {m: True for m in MODALITIES}
        pre, post = self._prompt(question or DEFAULT_QUESTION)
        mods = collate_mods_avail([sample], avail, self.device)
        sensor = self.proj(extract_tokens(self.alignment, mods, avail))
        embeds = batched_prompt_embeds(self.model.get_input_embeddings(),
                                       pre, post, sensor, self.device)
        texts = greedy_decode(self.model, embeds, self.tok,
                              max_new_tokens=max_new_tokens)
        text = texts[0]
        scores = match_scores(text, self.class_map)
        return {"text": text, "label": scores[0][0] if scores else -1,
                "class_name": self.class_map.get(scores[0][0], "") if scores else "",
                "top3": [(self.class_map.get(l, str(l)), round(s, 2)) for l, s in scores[:3]]}

    def sample_ids(self) -> list:
        return [s.id for s in self.samples]

    def get_sample(self, sid: str):
        for s in self.samples:
            if s.id == sid:
                return s
        raise KeyError(sid)

    @classmethod
    def from_ckpt(cls, ckpt_dir: str, dataset_root: str, anchors_path: str,
                  device: str = "cpu") -> "DemoEngine":
        from framework.llm_sft.eval_sft import load_sft_model
        model, proj, alignment, tok, cfg = load_sft_model(ckpt_dir, device)
        class_map = load_class_map(anchors_path)
        samples, missing, _ = load_split_base(dataset_root, "val", mode="lazy")
        return cls(model, proj, alignment, tok, samples, class_map, device)


class FakeEngine(DemoEngine):
    """Deterministic stub for UI smoke tests (no model download needed)."""

    def __init__(self):
        super().__init__(None, None, None, None, [], {0: "fake action"}, "cpu")

    def answer(self, sample, avail=None, question=None, max_new_tokens=16):
        off = [m for m in MODALITIES if not (avail or {}).get(m, True)]
        suffix = f" (modalities off: {','.join(off)})" if off else ""
        return {"text": f"A person is fake action.{suffix}",
                "label": 0, "class_name": "fake action", "top3": [("fake action", 1.0)]}

    def sample_ids(self):
        return ["FAKE_S01_A01_f1-5"]

    def get_sample(self, sid):
        import numpy as np
        from framework.dataset.sample import Modality, Sample
        mods = {m: Modality(data=np.random.rand(*s).astype("float32"),
                            frame_indices=list(range(s[0])), name=m)
                for m, s in TOY_DEMO_SHAPES.items()}
        return Sample(id=sid, label=0, modalities=mods, text={}, meta={})


TOY_DEMO_SHAPES = {"wifi": (5, 3, 16, 8), "depth": (5, 1, 32, 32),
                   "lidar": (5, 64, 3), "mmwave": (5, 64, 5), "rgb": (5, 17, 2)}


def make_engine(repo_root: str | None = None) -> DemoEngine:
    if os.environ.get("SFTMVP_DEMO_FAKE"):
        return FakeEngine()
    root = repo_root or os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    return DemoEngine.from_ckpt(
        ckpt_dir=os.path.join(root, "checkpoints_sftmvp"),
        dataset_root=os.path.join(root, "datasets", "mmfi", "v4"),
        anchors_path=os.path.join(root, "results", "sftmvp", "class_anchors.json"),
        device="cpu")
