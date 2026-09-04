"""Class anchor phrases from route-C captions + answer normalization/matching."""
from __future__ import annotations
import json
import re
from collections import Counter, defaultdict

_PERSON_RE = re.compile(r"^(a|an|the)\s+person\s+is\s+", re.IGNORECASE)
_STRIP_LEAD_RE = re.compile(r"^((?:(?:a|an|the)\s+)?person\s+is|(?:doing|answer))\s*[:,]?\s*", re.IGNORECASE)
_PUNCT_RE = re.compile(r"[^a-z0-9\s]+")


def build_class_map(captions_jsonl: str, num_classes: int = 27) -> dict[int, str]:
    """Majority-vote the first clause (before ':') of variants[0] per label."""
    votes: dict[int, Counter] = defaultdict(Counter)
    with open(captions_jsonl) as f:
        for line in f:
            d = json.loads(line)
            anchor = d["variants"][0].split(":")[0].strip().rstrip(".")
            anchor = _PERSON_RE.sub("", anchor).strip().lower()
            votes[int(d["label"])][anchor] += 1
    if len(votes) != num_classes:
        raise ValueError(f"expected {num_classes} classes, got {len(votes)}")
    return {k: c.most_common(1)[0][0] for k, c in sorted(votes.items())}


def load_class_map(path: str) -> dict[int, str]:
    raw = json.load(open(path))
    return {int(k): v for k, v in raw.items()}


def normalize(text: str) -> str:
    t = text.lower().strip()
    t = _PUNCT_RE.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = _STRIP_LEAD_RE.sub("", t).strip()
    return t


_SUFFIXES = ("ations", "ation", "ments", "ment", "ness", "ings", "ing", "ed", "es", "s")


def stem_tokens(text: str) -> list:
    out = []
    for w in normalize(text).split():
        for suf in _SUFFIXES:
            if len(w) > len(suf) + 2 and w.endswith(suf):
                w = w[: -len(suf)]
        out.append(w)
    return out


def match_answer(text: str, class_map: dict[int, str]) -> int:
    """Containment match first (exact rule), then stem-overlap >= 0.6 fallback
    (morphology-robust); longest overlap wins; -1 if none."""
    t = normalize(text)
    if not t:
        return -1
    best, best_score = -1, 0.0
    t_stems = stem_tokens(t)
    t_set = set(t_stems)
    for label, phrase in class_map.items():
        p = normalize(phrase)
        if not p:
            continue
        if p in t:
            score = len(p) + 100.0
        else:
            p_set = set(stem_tokens(p))
            inter = t_set & p_set
            if not inter:
                continue
            score = len(inter) / len(p_set)
            if score < 0.6:
                continue
        if score > best_score:
            best, best_score = label, score
    return best
