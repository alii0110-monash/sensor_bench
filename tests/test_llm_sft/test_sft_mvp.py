"""sftmvp unit tests: tiny Qwen2 config, real tokenizer (local), toy dataset."""
import json
import os
import pickle
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

MODEL_DIR = os.path.expanduser("~/models/qwen2.5-0.5b-instruct")

TOY_SHAPES = {"wifi": (5, 3, 16, 8), "depth": (5, 1, 32, 32),
              "lidar": (5, 64, 3), "mmwave": (5, 64, 5), "rgb": (5, 17, 2)}


def _toy_sample(sid: str, label: int):
    from framework.dataset.sample import Modality, Sample
    mods = {m: Modality(data=np.random.rand(*s).astype("float32"),
                        frame_indices=list(range(s[0])), name=m)
            for m, s in TOY_SHAPES.items()}
    return Sample(id=sid, label=label, modalities=mods, text={}, meta={})


def _toy_dataset(root: str, n_base: int = 6):
    data = os.path.join(root, "data")
    os.makedirs(data, exist_ok=True)
    os.makedirs(os.path.join(root, "splits"), exist_ok=True)
    train_ids, val_ids = [], []
    for i in range(n_base):
        sid = f"E01_S01_A{i % 3:02d}_f1-5"
        (train_ids if i < n_base - 2 else val_ids).append(sid)
        with open(os.path.join(data, f"{sid}.pkl"), "wb") as f:
            pickle.dump(_toy_sample(sid, i % 3).to_dict(), f)
    aug_id = f"E01_S01_A00_f1-5__aug_flip1"
    with open(os.path.join(data, f"{aug_id}.pkl"), "wb") as f:
        raw = _toy_sample(aug_id, 0).to_dict()
        raw["kind"] = "variant"
        raw["base_id"] = train_ids[0]
        raw["rgb"] = raw["modalities"]["rgb"]
        pickle.dump(raw, f)
    train_ids.append(aug_id)
    json.dump(train_ids, open(os.path.join(root, "splits", "train.json"), "w"))
    json.dump(val_ids, open(os.path.join(root, "splits", "val.json"), "w"))
    cap_path = os.path.join(root, "captions.jsonl")
    with open(cap_path, "w") as f:
        for sid in train_ids[:-1]:
            lbl = int(sid.split("A")[1][:2])
            f.write(json.dumps({"id": sid, "label": lbl,
                                "variants": [f"A person is class {lbl}: x", "y"]}) + "\n")
    return root, cap_path


@pytest.fixture(scope="module")
def tokenizer():
    if not os.path.isdir(MODEL_DIR):
        pytest.skip("Qwen tokenizer not downloaded")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_DIR)


@pytest.fixture(scope="function")
def tiny_model(tokenizer):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    cfg = Qwen2Config(vocab_size=len(tokenizer) + 8, hidden_size=64,
                      intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=1024)
    return Qwen2ForCausalLM(cfg)


def test_import_chain():
    import peft
    import transformers
    from transformers.models.qwen2 import modeling_qwen2
    assert peft.__version__ and transformers.__version__


def test_classmap_build_and_match(tmp_path):
    from framework.llm_sft.classmap import build_class_map, match_answer, normalize
    p = tmp_path / "cap.jsonl"
    votes = {0: ["jumping up"] * 5 + ["jumping downwards"],
             1: ["bowing"] * 3}
    with open(p, "w") as f:
        for lbl, phrases in votes.items():
            for ph in phrases:
                f.write(json.dumps({"id": f"x{lbl}{ph}", "label": lbl,
                                    "variants": [f"A person is {ph}: detail"]}) + "\n")
    cm = build_class_map(str(p), num_classes=2)
    assert cm[0] == "jumping up" and cm[1] == "bowing"
    assert normalize("A Person is: Bowing.") == "bowing"
    assert match_answer("the person is bowing", cm) == 1
    assert match_answer("A person is jumping up fast", cm) == 0
    assert match_answer("I have no idea", cm) == -1


def test_match_answer_morphology():
    from framework.llm_sft.classmap import match_answer, stem_tokens
    cm = {0: "stretching and relaxing", 1: "marking time in place"}
    assert match_answer("The person bends their arms for stretching and "
                        "relaxation", cm) == 0
    assert match_answer("He is marking time in the room", cm) == 1
    assert match_answer("completely unrelated words", cm) == -1


def test_stemmer_anchor_distinctness():
    from framework.llm_sft.classmap import load_class_map, stem_tokens
    cm = load_class_map(os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))),
        "results", "sftmvp", "class_anchors.json"))
    stems = {lbl: tuple(stem_tokens(p)) for lbl, p in cm.items()}
    vals = list(stems.values())
    assert len(set(vals)) == len(vals), "stemming collapsed distinct anchors"


def test_answer_for_step_rotation():
    from framework.llm_sft.train_sft import load_class_variants, answer_for_step
    cm = {0: "stretching and relaxing"}
    variants = {0: ["A person is stretching and relaxing.",
                    "The individual bends their arms for stretching and "
                    "relaxation; the tempo is brisk."]}
    got = {answer_for_step(0, f"s{i}", 0, variants) for i in range(20)}
    assert len(got) == 2 and any(g.startswith("A person is") for g in got)


def test_load_split_base(tmp_path):
    from framework.llm_sft.dataset import load_caption_ids, load_split_base
    root, cap_path = _toy_dataset(str(tmp_path))
    cap_ids = load_caption_ids(cap_path)
    samples, missing, pre = load_split_base(root, "train", caption_ids=cap_ids)
    assert len(samples) == 4 and len(missing) == 0
    assert all("__aug" not in s.id for s in samples)
    assert pre["n_selected"] == 4 and pre["fits"]
    vs, vmiss, _ = load_split_base(root, "val")
    assert len(vs) == 2


def test_projector_shapes():
    import torch
    from framework.llm_sft.projector import SensorTokenProjector
    proj = SensorTokenProjector(256, 64)
    out = proj(torch.randn(2, 5, 16, 256))
    assert out.shape == (2, 80, 64)


def test_sequence_and_padding(tokenizer, tiny_model):
    import torch
    from framework.llm_sft.prompting import (build_train_sample, encode_prompt_ids,
                                             encode_target_ids, pad_train_batch)
    emb = tiny_model.get_input_embeddings()
    pre, post = encode_prompt_ids(tokenizer)
    t = encode_target_ids(tokenizer, "A person is bowing.")
    se = torch.randn(80, 64)
    pad_emb = torch.zeros(64)
    e, lab, n_sensor = build_train_sample(emb, pre, post, t, se, "cpu", pad_emb)
    assert n_sensor == 80
    assert e.shape[0] == len(pre) + 80 + len(post) + len(t)
    assert (lab[:len(pre) + 80 + len(post)] == -100).all()
    assert (lab[-len(t):] == torch.tensor(t)).all()
    e2, lab2, _ = build_train_sample(emb, pre, post, t, None, "cpu", pad_emb)
    assert e2.shape[0] == len(pre) + len(post) + len(t)
    embeds, attn, labels = pad_train_batch([(e, lab, 80), (e2, lab2, 0)], pad_emb)
    assert embeds.shape[0] == 2 and attn.sum() == e.shape[0] + e2.shape[0]


def test_prompt_length_uniformity(tokenizer):
    from framework.llm_sft.prompting import encode_prompt_ids
    pre, post = encode_prompt_ids(tokenizer)
    assert len(pre) > 0 and len(post) > 0
    assert "<|im_start|>assistant" in tokenizer.decode(post)


def test_greedy_decode_runs(tokenizer, tiny_model):
    import torch
    from framework.llm_sft.prompting import (batched_prompt_embeds,
                                             encode_prompt_ids, greedy_decode)
    tiny_model.eval()
    pre, post = encode_prompt_ids(tokenizer)
    sensor = torch.randn(2, 80, 64)
    embeds = batched_prompt_embeds(tiny_model.get_input_embeddings(),
                                   pre, post, sensor, "cpu")
    assert embeds.shape[0] == 2 and embeds.shape[1] == len(pre) + 80 + len(post)
    texts = greedy_decode(tiny_model, embeds, tokenizer, max_new_tokens=4)
    assert len(texts) == 2 and all(isinstance(t, str) for t in texts)


def test_train_step_decreases(tokenizer, tiny_model, tmp_path):
    import torch
    from framework.llm_sft.dataset import collate_mods
    from framework.llm_sft.projector import SensorTokenProjector, extract_tokens
    from framework.llm_sft.prompting import (build_train_sample, encode_prompt_ids,
                                             encode_target_ids, pad_train_batch)
    from framework.llm_sft.train_sft import answer_for, build_lora

    from framework.models.alignment import AlignmentModel
    alignment = AlignmentModel(text_dim=512).eval()
    for p in alignment.parameters():
        p.requires_grad_(False)

    for p in tiny_model.parameters():
        p.requires_grad_(False)
    model = build_lora(tiny_model, r=4, alpha=8)
    model.train()
    proj = SensorTokenProjector(256, 64)
    class_map = {i: f"action {i}" for i in range(3)}
    samples = [_toy_sample(f"s{i}", i % 3) for i in range(6)]
    pre, post = encode_prompt_ids(tokenizer)
    emb_layer = model.get_input_embeddings()
    pad_emb = emb_layer(torch.tensor(tokenizer.pad_token_id
                                     or tokenizer.eos_token_id))
    opt = torch.optim.AdamW([{"params": proj.parameters(), "lr": 1e-3},
                             {"params": [p for p in model.parameters()
                                         if p.requires_grad], "lr": 1e-3}])
    losses = []
    for _ in range(20):
        batch = samples[:4]
        mods, _ = collate_mods(batch, "cpu")
        sensor = proj(extract_tokens(alignment, mods))
        rows = [build_train_sample(emb_layer, pre, post,
                                   encode_target_ids(tokenizer,
                                                     answer_for(s.label, class_map)),
                                   se, "cpu", pad_emb)
                for s, se in zip(batch, sensor)]
        embeds, attn, labels = pad_train_batch(rows, pad_emb)
        out = model(inputs_embeds=embeds, attention_mask=attn, labels=labels)
        opt.zero_grad()
        out.loss.backward()
        opt.step()
        losses.append(out.loss.item())
    assert all(np.isfinite(losses))
    assert losses[-1] < losses[0]


def test_adapter_roundtrip(tokenizer, tiny_model, tmp_path):
    import torch
    from peft import PeftModel
    from transformers import Qwen2ForCausalLM
    from framework.llm_sft.train_sft import build_lora
    base_sd = {k: v.clone() for k, v in tiny_model.state_dict().items()}
    model = build_lora(tiny_model, r=4, alpha=8)
    d = str(tmp_path / "adapter")
    model.save_pretrained(d)
    base2 = Qwen2ForCausalLM(tiny_model.config)
    base2.load_state_dict(base_sd)
    for p in base2.parameters():
        p.requires_grad_(False)
    reloaded = PeftModel.from_pretrained(base2, d)
    x = torch.randint(0, 100, (1, 8))
    with torch.no_grad():
        a = model(input_ids=x).logits
        b = reloaded(input_ids=x).logits
    assert torch.allclose(a, b, atol=1e-5)
