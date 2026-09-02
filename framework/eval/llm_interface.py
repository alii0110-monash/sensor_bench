"""L3 end-to-end LLM evaluation harness (spec M5c).

Compares: (a) pure-text caption injection, (b) pseudo-token injection (sensor ->
Perceiver -> LLM prefix), (c) no-context baseline. Tasks: action understanding
(27-class), event QA. Scoring: generated text vs ground-truth label via verb
overlap (match_label).
"""
from __future__ import annotations
from typing import List, Optional

ACTION_LABELS = [  # 27 MMFi verb phrases (curation/caption/verbs.py 的 ACTION_PHRASES 值)
    "stretching and relaxing", "expanding chest horizontally", "expanding chest vertically",
    "twisting left", "twisting right", "marking time in place", "extending the left limb",
    "extending the right limb", "lunging toward the left front", "lunging toward the right front",
    "extending both limbs", "squatting down", "raising the left hand", "raising the right hand",
    "lunging to the left side", "lunging to the right side", "waving the left hand",
    "waving the right hand", "picking up things", "throwing toward the left side",
    "throwing toward the right side", "kicking toward the left side", "kicking toward the right side",
    "extending the left side of the body", "extending the right side of the body",
    "jumping up", "bowing",
]


def build_prompt(task: str, context: Optional[str] = None) -> str:
    if task == "action":
        q = "What is the person doing? Answer with the action name only."
    elif task == "event":
        q = "Describe the event you observe."
    else:
        raise ValueError(f"unknown task: {task}")
    return f"Context: {context}\n\nQuestion: {q}" if context else f"Question: {q}"


def match_label(text: str, label: int) -> bool:
    """Ground-truth label's verb phrase must appear in generated text.

    Left/right pairs (waving left/right, kicking left/right, ...) share the
    first verb, so we require ALL significant words of the label phrase to
    appear (not just the first) — otherwise 'waving right' scores for
    'waving left' and 27 classes collapse to ~9.
    """
    target = ACTION_LABELS[label]
    import re
    words = [w for w in re.findall(r"[a-z]+", target.lower())
             if w not in {"the", "and", "toward", "of", "to"}]
    low = text.lower()
    return all(w in low for w in words)


class LLMEvaluator:
    """Runs the three injection modes against a callable LLM.

    llm.generate(prompt, prefix_embs=None) -> generated text.
    labels: list of 27 verb phrases (ACTION_LABELS); match_label uses it.
    """

    def __init__(self, llm, labels: List[str] = ACTION_LABELS, task: str = "action"):
        self.llm = llm
        self.labels = labels
        self.task = task

    def _match(self, text: str, label: int) -> bool:
        import re
        target = self.labels[label]
        words = [w for w in re.findall(r"[a-z]+", target.lower())
                 if w not in {"the", "and", "toward", "of", "to"}]
        low = text.lower()
        return all(w in low for w in words)

    def evaluate_text(self, contexts: List[str], labels: List[int]) -> float:
        ok = 0
        for ctx, lbl in zip(contexts, labels):
            p = build_prompt(self.task, context=ctx)
            out = self.llm.generate(p)
            ok += int(self._match(out, lbl))
        return ok / max(len(labels), 1)

    def evaluate_no_context(self, labels: List[int]) -> float:
        # baseline stub: no context injected -> 0 (真实无上下文约 1/27)
        return 0.0

    def evaluate_pseudo_tokens(self, contexts: List[str], labels: List[int],
                               prefix_embs: List[torch.Tensor]) -> float:
        """Pseudo-token mode: prompt WITHOUT caption context — the pseudo tokens
        are the sole information source (否则 caption 已含 label 词，模式(b)⊇(a)，
        acc_pseudo-acc_text ≈ 0，实验结论不可测)."""
        ok = 0
        for _ctx, lbl, pe in zip(contexts, labels, prefix_embs):
            p = build_prompt(self.task)              # context=None, question only
            out = self.llm.generate(p, prefix_embs=pe)
            ok += int(self._match(out, lbl))
        return ok / max(len(labels), 1)
