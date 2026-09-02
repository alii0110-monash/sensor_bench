"""L3 harness unit tests with a fake LLM (no real load)."""
import torch
import pytest
from framework.eval.llm_interface import LLMEvaluator, build_prompt, match_label

class FakeLLM:
    """Deterministic fake: echoes 'action A01' style, counts calls."""
    def __init__(self):
        self.calls = []
    def generate(self, prompt, prefix_embs=None):
        self.calls.append((prompt, prefix_embs))
        return "the person is stretching and relaxing"

def test_build_prompt_action():
    p = build_prompt(task="action")
    assert "doing" in p.lower() and "action" in p.lower()

def test_build_prompt_with_context():
    p = build_prompt(task="action", context="In the video, a man is stretching.")
    assert "In the video, a man is stretching." in p

def test_match_label():
    # "stretching" matches A01 phrase "stretching and relaxing"
    assert match_label("the person is stretching and relaxing", label=0)  # A01
    assert not match_label("the person is waving", label=0)
    # left/right distinction: waving left (label 16) vs waving right (label 17)
    assert match_label("the person is waving the left hand", label=16)
    assert not match_label("the person is waving the right hand", label=16)
    assert match_label("the person is waving the right hand", label=17)
    assert not match_label("the person is waving the left hand", label=17)

def test_evaluator_text_only():
    llm = FakeLLM()
    ev = LLMEvaluator(llm, labels=["stretching and relaxing", "waving"])
    acc = ev.evaluate_text(["In the video, a man is stretching."], [0])
    assert acc == 1.0
    assert len(llm.calls) == 1

def test_evaluator_no_context_baseline():
    llm = FakeLLM()
    ev = LLMEvaluator(llm, labels=["stretching and relaxing", "waving"])
    acc = ev.evaluate_no_context([0])  # no prompt, baseline
    assert acc == 0.0  # nothing to match without context


def test_action_verb_anchor_distinct():
    """动作词锚: 不同动作的 llama2 词 embedding 应可区分 (蒸馏目标有效)."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    from train_projection import _action_verb_emb
    from framework.models.llm_adapter import LlamaAdapter
    adapter = LlamaAdapter(k=8, device="cpu")
    adapter._load()  # 加载 llama2 (慢, 但语义锚需要真实 embedding)
    emb = _action_verb_emb(adapter, adapter._tokenizer, [0, 16, 17], "cpu")  # stretching, waving left, waving right
    sim = emb @ emb.t()
    # 不同动作应可区分 (sim 明显 < 1)
    assert sim[0, 1] < 0.95, f"stretching vs waving 应可区分 (sim={sim[0,1]:.3f})"
    assert emb.shape == (3, 4096)
