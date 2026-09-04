#!/usr/bin/env python
"""Yes/no probe eval: can the model confirm/correct a claimed action based on
sensor tokens? Positive probe (gold) expects Yes; negative probe (wrong class)
expects No + correction. Complements the frozen default-question eval."""
import argparse, json, os, random, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from framework.llm_sft.classmap import normalize
from framework.llm_sft.eval_sft import load_sft_model
from framework.llm_sft.dataset import load_split_base
from framework.llm_sft.prompting import batched_prompt_embeds, greedy_decode
from framework.llm_sft.projector import extract_tokens
from framework.models.alignment import MODALITIES


@torch.no_grad()
def run_probe(model, proj, alignment, tok, samples, class_map, device,
              batch_size=16, n=200, seed=0):
    from framework.llm_sft.prompting import encode_prompt_ids
    rng = random.Random(seed)
    subset = rng.sample(samples, min(n, len(samples)))
    emb_layer = model.get_input_embeddings()
    res = {"yes": [0, 0], "no": [0, 0]}
    gens = []
    for i in range(0, len(subset), batch_size):
        batch = subset[i:i + batch_size]
        mods = {}
        import numpy as np
        for m in MODALITIES:
            mods[m] = torch.from_numpy(np.stack(
                [s.modalities[m].data for s in batch]).astype("float32")).to(device)
        sensor = proj(extract_tokens(alignment, mods))
        for s, se in zip(batch, sensor):
            gold_anchor = class_map[s.label]
            # positive: claim gold -> expect yes; negative: claim wrong -> expect no
            claim = gold_anchor
            q = f"Is the person {claim}?"
            pre, post = encode_prompt_ids(tok, q)
            e2 = batched_prompt_embeds(emb_layer, pre, post, se.unsqueeze(0), device)
            t2 = greedy_decode(model, e2, tok, max_new_tokens=14)[0]
            n2 = normalize(t2)
            is_yes = n2.startswith("yes") or "yes" in n2[:12]
            res["yes"][0] += int(is_yes)
            res["yes"][1] += 1
            gens.append({"id": s.id, "probe": "pos", "gen": t2, "hit": is_yes})
            wrong = rng.choice([l for l in class_map if l != s.label])
            q = f"Is the person {class_map[wrong]}?"
            pre, post = encode_prompt_ids(tok, q)
            e3 = batched_prompt_embeds(emb_layer, pre, post, se.unsqueeze(0), device)
            t3 = greedy_decode(model, e3, tok, max_new_tokens=14)[0]
            n3 = normalize(t3)
            is_no = n3.startswith("no") or ("no," in n3[:12]) or ("actually" in n3[:20])
            res["no"][0] += int(is_no)
            res["no"][1] += 1
            gens.append({"id": s.id, "probe": "neg", "gen": t3, "hit": is_no})
    return res, gens


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints_sftmvp_v3")
    ap.add_argument("--dataset", default="datasets/mmfi/v4")
    ap.add_argument("--anchors", default="results/sftmvp/class_anchors.json")
    ap.add_argument("--out", default="results/sftmvp/eval_v3_yesno.json")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, proj, alignment, tok, cfg = load_sft_model(args.ckpt, device)
    class_map = {int(k): v for k, v in json.load(open(args.anchors)).items()}
    samples, missing, _ = load_split_base(args.dataset, "val", mode="lazy")
    res, gens = run_probe(model, proj, alignment, tok, samples, class_map,
                          device, batch_size=args.batch_size, n=args.n,
                          seed=args.seed)
    out = {"ckpt": args.ckpt, "n_probes": res["yes"][1] + res["no"][1],
           "acc_yes": res["yes"][0] / max(res["yes"][1], 1),
           "acc_no": res["no"][0] / max(res["no"][1], 1),
           "acc_total": (res["yes"][0] + res["no"][0]) /
                        max(res["yes"][1] + res["no"][1], 1)}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    with open(args.out.replace(".json", "_generations.jsonl"), "w") as f:
        for g in gens:
            f.write(json.dumps(g) + "\n")
    print(f"[yesno] acc_yes={out['acc_yes']:.3f} acc_no={out['acc_no']:.3f} "
          f"total={out['acc_total']:.3f}", flush=True)


if __name__ == "__main__":
    main()
