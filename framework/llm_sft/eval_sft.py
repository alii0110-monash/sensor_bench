"""Paired evaluation: with-token vs text-only, greedy decode + anchor matching."""
from __future__ import annotations
import json
import os
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from framework.llm_sft.classmap import load_class_map, match_answer
from framework.llm_sft.dataset import collate_mods, load_split_base
from framework.llm_sft.projector import SensorTokenProjector, extract_tokens, load_frozen_encoders
from framework.llm_sft.prompting import (batched_prompt_embeds, encode_prompt_ids,
                                         greedy_decode)


def load_sft_model(ckpt_dir: str, device: str):
    cfg = json.load(open(os.path.join(ckpt_dir, "run_config.json")))
    tok = AutoTokenizer.from_pretrained(os.path.join(ckpt_dir, "adapter"))
    base = AutoModelForCausalLM.from_pretrained(
        cfg["model_dir"], torch_dtype=torch.float32)
    model = PeftModel.from_pretrained(base, os.path.join(ckpt_dir, "adapter"))
    model.config.use_cache = True
    model = model.to(device).eval()
    proj = SensorTokenProjector(256, cfg["hidden_size"]).to(device)
    proj.load_state_dict(torch.load(os.path.join(ckpt_dir, "projector.pt"),
                                    map_location="cpu", weights_only=True))
    proj.eval()
    alignment = load_frozen_encoders(cfg["encoders_ckpt"], device)
    return model, proj, alignment, tok, cfg


@torch.no_grad()
def run_condition(model, proj, alignment, tok, samples, pre_ids, post_ids,
                  device, batch_size: int = 16, with_sensor: bool = True,
                  max_new_tokens: int = 12, collect_gens: int = 0):
    emb_layer = model.get_input_embeddings()
    preds, gens = [], []
    t0 = time.time()
    for i in range(0, len(samples), batch_size):
        batch = list(samples[i:i + batch_size])
        sensor_b = None
        if with_sensor:
            mods, _ = collate_mods(batch, device)
            sensor_b = proj(extract_tokens(alignment, mods))
        embeds = batched_prompt_embeds(emb_layer, pre_ids, post_ids, sensor_b, device)
        texts = greedy_decode(model, embeds, tok, max_new_tokens=max_new_tokens)
        preds.extend(texts)
        if collect_gens and len(gens) < collect_gens:
            for s, t in zip(batch, texts):
                gens.append({"id": s.id, "label": s.label, "gen": t})
    sec = time.time() - t0
    return preds, gens, sec


def evaluate(ckpt_dir: str, dataset_root: str, anchors_path: str, out_path: str,
             device: str = "cuda", batch_size: int = 16, load_mode: str = "auto",
             max_new_tokens: int = 12, collect_gens: int = 200,
             limit: int = 0) -> dict:
    model, proj, alignment, tok, cfg = load_sft_model(ckpt_dir, device)
    class_map = load_class_map(anchors_path)
    samples, missing, pre = load_split_base(dataset_root, "val", mode=load_mode)
    if limit > 0:
        samples = samples[:limit]
    print(f"[sftmvp-eval] val base={len(samples)} missing={len(missing)} "
          f"mode={pre['mode']}", flush=True)
    pre_ids, post_ids = encode_prompt_ids(tok)

    results = {"n": len(samples), "val_missing": len(missing),
               "val_missing_ids": missing, "preflight": pre, "ckpt": ckpt_dir}
    gens_all = {}
    for cond, with_sensor in (("with_token", True), ("text_only", False)):
        texts, gens, sec = run_condition(model, proj, alignment, tok, samples,
                                         pre_ids, post_ids, device,
                                         batch_size=batch_size, with_sensor=with_sensor,
                                         max_new_tokens=max_new_tokens,
                                         collect_gens=collect_gens)
        matched = [match_answer(t, class_map) for t in texts]
        labels = [s.label for s in samples]
        acc = sum(m == l for m, l in zip(matched, labels)) / max(len(labels), 1)
        unmatched = sum(m == -1 for m in matched)
        per_class = {}
        for l, m in zip(labels, matched):
            c = per_class.setdefault(l, [0, 0])
            c[0] += int(m == l)
            c[1] += 1
        results[cond] = {"acc": acc, "n_unmatched": unmatched,
                         "per_class_acc": {str(k): v[0] / v[1] for k, v in sorted(per_class.items())},
                         "sec": round(sec, 1)}
        gens_all[cond] = gens
        print(f"[sftmvp-eval] {cond}: acc={acc:.4f} unmatched={unmatched} ({sec}s)",
              flush=True)

    results["delta"] = results["with_token"]["acc"] - results["text_only"]["acc"]
    results["positive"] = bool(results["with_token"]["acc"] >= 0.074 and
                               results["delta"] >= 0.05)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(results, open(out_path, "w"), indent=1)
    gens_path = out_path.replace(".json", "_generations.jsonl")
    with open(gens_path, "w") as f:
        for cond in gens_all:
            for g in gens_all[cond]:
                f.write(json.dumps({"cond": cond, **g}) + "\n")
    print(f"[sftmvp-eval] acc_pseudo={results['with_token']['acc']:.4f} "
          f"acc_text={results['text_only']['acc']:.4f} delta={results['delta']:+.4f} "
          f"positive={results['positive']}", flush=True)
    return results
