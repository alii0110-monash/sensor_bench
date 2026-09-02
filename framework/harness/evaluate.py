from __future__ import annotations
from typing import Dict, List


def accuracy(preds, labels) -> float:
    if not labels:
        return 0.0
    ok = 0
    for p, l in zip(preds, labels):
        pred = max(p, key=p.get) if isinstance(p, dict) else int(p)
        ok += pred == l
    return ok / len(labels)


def evaluate_model(model, samples, profile: dict, batch_size: int = 64) -> dict:
    preds = []
    if hasattr(model, "predict_batch"):
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            logits = model.predict_batch(batch, profile["available"])
            preds += [max(range(logits.shape[1]), key=lambda c: logits[j, c].item())
                      for j in range(logits.shape[0])]
    else:
        preds = [model.predict(s, profile["available"]) for s in samples]
    labels = [s.label for s in samples]
    return {"profile": profile["id"], "available": profile["available"],
            "accuracy": accuracy(preds, labels)}
