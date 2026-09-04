"""LoRA SFT training: projector + LoRA on frozen Qwen2.5-0.5B-Instruct."""
from __future__ import annotations
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from framework.llm_sft.classmap import load_class_map
from framework.llm_sft.dataset import collate_mods, load_caption_ids, load_split_base
from framework.llm_sft.projector import SensorTokenProjector, extract_tokens, load_frozen_encoders
from framework.llm_sft.prompting import (build_train_sample, encode_prompt_ids,
                                         encode_target_ids, pad_train_batch)


def build_lora(model, r: int = 16, alpha: int = 32, dropout: float = 0.05):
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout,
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                     bias="none", task_type="CAUSAL_LM")
    return get_peft_model(model, cfg)


def answer_for(label: int, class_map: dict) -> str:
    return f"A person is {class_map[label]}."


QA_WEIGHTS = {"what": 0.4, "yesno": 0.4, "describe": 0.2}
DESCRIBE_Q = "Describe the motion you observe."


def build_qa(label: int, sid: str, epoch: int, class_map: dict, rng) -> tuple:
    """Question-conditioned (question, answer) pair; formats stay anchor-
    contained so the frozen matcher keeps working."""
    anchor = class_map[label]
    r = rng.random()
    if r < QA_WEIGHTS["what"]:
        return None, f"A person is {anchor}."
    if r < QA_WEIGHTS["what"] + QA_WEIGHTS["yesno"]:
        wrong = rng.choice([l for l in class_map if l != label])
        if rng.random() < 0.5:
            return f"Is the person {class_map[wrong]}?", f"No, actually the person is {anchor}."
        return f"Is the person {anchor}?", f"Yes, the person is {anchor}."
    return DESCRIBE_Q, f"The person appears to be {anchor}."


def load_class_variants(captions_jsonl: str, class_map: dict) -> dict[int, list]:
    """Per-class list of [canonical anchor, variant sentences (<=24 words)...]
    from a C2-style captions jsonl {id,label,variants}."""
    import json as _json
    by_class: dict[int, list] = {}
    with open(captions_jsonl) as f:
        for line in f:
            d = _json.loads(line)
            lst = by_class.setdefault(int(d["label"]), [])
            for v in d["variants"]:
                words = v.split()
                if len(words) > 24:
                    v = " ".join(words[:24])
                if v and v not in lst:
                    lst.append(v)
    for lbl, anchor in class_map.items():
        lst = by_class.setdefault(lbl, [f"A person is {anchor}."])
        if f"A person is {anchor}." not in lst:
            lst.insert(0, f"A person is {anchor}.")
    return by_class


def answer_for_step(label: int, sid: str, epoch: int, variants: dict[int, list]) -> str:
    """Rotate per-sample through the class's answer list (slot 0 = canonical
    anchor, rest = C2 variant sentences) — supervision width with a stable anchor."""
    opts = variants[label]
    phase = sum(ord(c) for c in sid) % len(opts)
    return opts[(epoch + phase) % len(opts)]


@torch.no_grad()
def val_loss(model, proj, alignment, tok, samples, pre_ids, post_ids,
             class_map, device, max_samples: int = 500, batch_size: int = 32) -> float:
    model.eval()
    emb_layer = model.get_input_embeddings()
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.convert_tokens_to_ids("<|im_end|>")
    pad_emb = emb_layer(torch.tensor(pad_id, device=device))
    subset = samples[:max_samples]
    total, n = 0.0, 0
    for i in range(0, len(subset), batch_size):
        batch = list(subset[i:i + batch_size])
        mods, _ = collate_mods(batch, device)
        sensor = proj(extract_tokens(alignment, mods))
        rows = []
        for s, se in zip(batch, sensor):
            t_ids = encode_target_ids(tok, answer_for(s.label, class_map))
            rows.append(build_train_sample(emb_layer, pre_ids, post_ids, t_ids,
                                           se, device, pad_emb))
        embeds, attn, labels = pad_train_batch(rows, pad_emb)
        out = model(inputs_embeds=embeds, attention_mask=attn, labels=labels)
        total += out.loss.item() * len(batch)
        n += len(batch)
    model.train()
    return total / max(n, 1)


def train(dataset_root: str, encoders_ckpt: str, captions_jsonl: str,
          anchors_path: str, model_dir: str, out_dir: str,
          epochs: int = 4, batch_size: int = 32, seed: int = 0,
          lr_proj: float = 1e-3, lr_lora: float = 1e-4,
          device: str = "cuda", load_mode: str = "auto",
          max_train: int = 0, log_every: int = 20, log_path: str = "",
          variants_jsonl: str = "", qa_multi: bool = False) -> dict:
    torch.manual_seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(model_dir)
    base = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
    base.config.use_cache = False
    for p in base.parameters():
        p.requires_grad_(False)
    model = build_lora(base)
    proj = SensorTokenProjector(256, model.config.hidden_size).to(device)
    model = model.to(device)
    alignment = load_frozen_encoders(encoders_ckpt, device)
    model.train()

    class_map = load_class_map(anchors_path)
    if variants_jsonl:
        class_variants = load_class_variants(variants_jsonl, class_map)
        n_opts = sum(len(v) for v in class_variants.values()) / max(len(class_variants), 1)
        print(f"[sftmvp] answer variants: {n_opts:.1f}/class (supervision width on)",
              flush=True)
    else:
        class_variants = {lbl: [f"A person is {a}."] for lbl, a in class_map.items()}

    def _answer(s, ep: int) -> str:
        return answer_for_step(s.label, s.id, ep, class_variants)

    if qa_multi:
        import random as _random
        qa_rng = _random.Random(seed)
        qa_cache = {}

        def _qa_ids(s, ep: int):
            q, ans = build_qa(s.label, s.id, ep, class_map, qa_rng)
            if q not in qa_cache:
                qa_cache[q] = encode_prompt_ids(tok, q)
            pre, post = qa_cache[q]
            return pre, post, encode_target_ids(tok, ans)
    else:
        def _qa_ids(s, ep: int):
            return None

    cap_ids = load_caption_ids(captions_jsonl)
    train_samples, train_missing, pre_train = load_split_base(
        dataset_root, "train", mode=load_mode, caption_ids=cap_ids)
    val_samples, val_missing, pre_val = load_split_base(dataset_root, "val", mode=load_mode)
    print(f"[sftmvp] train base={len(train_samples)} (missing {len(train_missing)}) "
          f"val base={len(val_samples)} (missing {len(val_missing)})", flush=True)
    if max_train > 0:
        train_samples = train_samples[:max_train]
        print(f"[sftmvp] max_train={max_train} -> {len(train_samples)}", flush=True)

    pre_ids, post_ids = encode_prompt_ids(tok)
    emb_layer = model.get_input_embeddings()
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.convert_tokens_to_ids("<|im_end|>")
    pad_emb = emb_layer(torch.tensor(pad_id, device=device))

    proj_params = list(proj.parameters())
    lora_params = [p for n, p in model.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": proj_params, "lr": lr_proj},
        {"params": lora_params, "lr": lr_lora},
    ])

    log = {"args": {k: v for k, v in locals().items() if isinstance(v, (int, float, str, bool))},
           "preflight": {"train": pre_train, "val": pre_val},
           "n_train": len(train_samples), "n_val": len(val_samples),
           "epochs": [], "steps": []}
    g = torch.Generator().manual_seed(seed)
    step = 0
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(len(train_samples), generator=g).tolist()
        ep_loss, ep_n = 0.0, 0
        for i in range(0, len(perm), batch_size):
            batch = [train_samples[j] for j in perm[i:i + batch_size]]
            mods, _ = collate_mods(batch, device)
            sensor = proj(extract_tokens(alignment, mods))
            rows = []
            for s, se in zip(batch, sensor):
                t_ids = encode_target_ids(tok, _answer(s, ep))
                pre_s, post_s = pre_ids, post_ids
                if qa_multi:
                    pre_s, post_s, t_ids = _qa_ids(s, ep)
                rows.append(build_train_sample(emb_layer, pre_s, post_s, t_ids,
                                               se, device, pad_emb))
            embeds, attn, labels = pad_train_batch(rows, pad_emb)
            out = model(inputs_embeds=embeds, attention_mask=attn, labels=labels)
            opt.zero_grad()
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in proj_params + lora_params if p.grad is not None], 1.0)
            opt.step()
            ep_loss += out.loss.item() * len(batch)
            ep_n += len(batch)
            step += 1
            if step % log_every == 0:
                sps = step / (time.time() - t0)
                print(f"[sftmvp] ep{ep} step{step} loss={out.loss.item():.4f} "
                      f"({sps:.1f} steps/s)", flush=True)
                log["steps"].append({"ep": ep, "step": step, "loss": out.loss.item()})
        vl = val_loss(model, proj, alignment, tok, val_samples, pre_ids, post_ids,
                      class_map, device)
        entry = {"epoch": ep, "train_loss": ep_loss / max(ep_n, 1), "val_loss": vl,
                 "sec": round(time.time() - t0, 1)}
        log["epochs"].append(entry)
        print(f"[sftmvp] epoch {ep}: train {entry['train_loss']:.4f} "
              f"val {vl:.4f} ({entry['sec']}s)", flush=True)

    torch.save(proj.state_dict(), os.path.join(out_dir, "projector.pt"))
    adapter_dir = os.path.join(out_dir, "adapter")
    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    json.dump({"dataset": dataset_root, "encoders_ckpt": encoders_ckpt,
               "anchors": anchors_path, "model_dir": model_dir,
               "hidden_size": model.config.hidden_size, "seed": seed,
               "epochs": epochs, "batch_size": batch_size,
               "qa_multi": qa_multi, "variants_jsonl": variants_jsonl},
              open(os.path.join(out_dir, "run_config.json"), "w"), indent=1)
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        json.dump(log, open(log_path, "w"), indent=1)
    return log
