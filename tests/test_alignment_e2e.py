# tests/test_alignment_e2e.py
import json, os, pickle, sys
import numpy as np
import torch
import pytest
from framework.dataset.loader import load_dataset
from framework.models.alignment import AlignmentModel
from framework.models.text_encoder import HashTextEncoder
from framework.dataset.sample import Sample, Modality

def _mini_v5(tmp_path, n=8):
    root = tmp_path / "v5"
    (root / "data").mkdir(parents=True)
    ids = []
    for i in range(n):
        mm = {
            "wifi": Modality(data=np.random.randn(2, 3, 4, 4).astype(np.float32),  # (T,C,H,W)
                             frame_indices=[1, 2], sample_rate=10),
            "rgb": Modality(data=np.random.randn(2, 17, 2).astype(np.float32),     # (T,P,C)
                            frame_indices=[1, 2], sample_rate=10),
        }
        s = Sample(id=f"s{i}", label=i % 3, modalities=mm,
                   text={"en": [f"action number {i % 3}"]})
        with open(root / "data" / f"{s.id}.pkl", "wb") as f:
            pickle.dump(s.to_dict(), f)
        ids.append(s.id)
    (root / "splits").mkdir(exist_ok=True)
    json.dump(ids[:6], open(root / "splits" / "train.json", "w"))
    json.dump(ids[6:], open(root / "splits" / "test.json", "w"))
    json.dump([], open(root / "splits" / "val.json", "w"))
    with open(root / "modalities.yaml", "w") as f:
        f.write("modalities:\n- wifi\n- rgb\n")
    return root

def test_train_alignment_mini(tmp_path):
    root = _mini_v5(tmp_path)
    ds = load_dataset(str(root))
    m = AlignmentModel(num_modalities=5, text_dim=512, num_classes=27)
    te = HashTextEncoder(dim=512)
    cfg = {"epochs": 2, "batch_size": 4, "lr": 1e-3, "device": "cpu"}
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    from train_alignment import train_epoch
    opt = torch.optim.AdamW(m.parameters(), lr=cfg["lr"])
    p0 = list(m.parameters())[0].detach().clone()
    loss, nce, ce = train_epoch(m, te, ds.train, opt, batch_size=4, device="cpu",
                                aux_cls_weight=0.5, neg_mine=True)
    assert loss > 0 and torch.isfinite(torch.tensor(loss))
    assert torch.isfinite(torch.tensor(nce)) and torch.isfinite(torch.tensor(ce))
    # model params updated
    assert not torch.allclose(p0, list(m.parameters())[0].detach())


def test_checkpoint_roundtrip_strips_cls_head(tmp_path):
    from framework.models.alignment import AlignmentModel
    m = AlignmentModel(num_modalities=5, text_dim=512, num_classes=27)
    state = m.state_dict()
    assert any(k.startswith("classification_head.") for k in state)
    stripped = {k: v for k, v in state.items()
                if not k.startswith("classification_head.")}
    ev = AlignmentModel(num_modalities=5, text_dim=512)   # eval 默认无分类头
    ev.load_state_dict(stripped)                          # strict=True 应通过


import numpy as np
from framework.eval.alignment import build_held_out_split, retrieval_recall_at_k

def test_build_held_out_split():
    bases = [f"E01_S01_A01_f{i}-{i+6}" for i in range(20)]
    variants = [f"{b}__aug{k}" for b in bases[:10] for k in range(4)]
    all_ids = bases + variants
    held, train_ids = build_held_out_split(all_ids, fraction=0.1, seed=0)
    # held contains whole base groups (base + its variants)
    held_bases = {x.split("__")[0] for x in held}
    assert 0 < len(held_bases) <= 3  # ~10% of 20
    assert all(x not in train_ids for x in held)

def test_retrieval_recall_at_k():
    # 10 queries, embeddings where nearest neighbor is correct
    rng = np.random.default_rng(0)
    q = rng.standard_normal((10, 8)); t = q.copy()  # exact matches
    r1 = retrieval_recall_at_k(torch.tensor(q), torch.tensor(t), k=1)
    assert r1 == 1.0
    # reverse direction: text query -> sensor candidate
    rt1 = retrieval_recall_at_k(torch.tensor(t), torch.tensor(q), k=1)
    assert rt1 == 1.0
    # random -> ~0 recall@1
    t2 = torch.tensor(rng.standard_normal((10, 8)))
    r2 = retrieval_recall_at_k(torch.tensor(q), t2, k=1)
    assert r2 < 0.3
