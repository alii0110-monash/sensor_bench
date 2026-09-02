"""Synthetic caption generation for MMFi samples.

TemplateCaptioner is deterministic (no LLM call) — used for tests and as a
fallback. The real pipeline may subclass SyntheticCaptioner with an LLM backend
(local/API), keeping the `generate(sample) -> List[str]` interface.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List

from .verbs import LABEL_TO_VERB


class SyntheticCaptioner(ABC):
    """Generate 3-5 natural-language sentences describing one sample."""

    @abstractmethod
    def generate(self, sample: Dict) -> List[str]:
        ...


class TemplateCaptioner(SyntheticCaptioner):
    """Deterministic template captions anchored on action verb + metadata.

    Sentence templates are cycled for diversity (n sentences from n patterns).
    """

    _TEMPLATES = [
        "A person is {verb}.",
        "We observe a person {verb}.",
        "The subject can be seen {verb}.",
        "In this scene, someone is {verb}.",
        "This clip shows a person {verb}.",
    ]

    def __init__(self, n: int = 3):
        self.n = n

    def generate(self, sample: Dict) -> List[str]:
        verb = LABEL_TO_VERB(sample["label"])
        meta = sample.get("meta", {})
        env = meta.get("env", "")
        subj = meta.get("subject", "")
        out = []
        for i in range(self.n):
            t = self._TEMPLATES[i % len(self._TEMPLATES)].format(verb=verb)
            if env:
                t = f"{t} Environment: {env}."
            if subj:
                t = f"{t} Subject: {subj}."
            out.append(t)
        return out
