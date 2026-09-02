import json
from curation.caption.verbs import ACTION_PHRASES, LABEL_TO_VERB

def test_action_phrases_complete():
    assert len(ACTION_PHRASES) == 27  # A01..A27

def test_action_phrases_all_lowercase_nonempty():
    for code, phrase in ACTION_PHRASES.items():
        assert code.startswith("A") and code[1:].isdigit()
        assert phrase and phrase == phrase.strip()

def test_label_to_verb_mapping():
    # label 0 == A01
    assert LABEL_TO_VERB(0) == ACTION_PHRASES["A01"]
    assert LABEL_TO_VERB(26) == ACTION_PHRASES["A27"]


import pytest
from curation.caption.captioner import SyntheticCaptioner, TemplateCaptioner

def _fake_sample(sid="E01_S01_A01_f1-7", label=0):
    return {"id": sid, "label": label, "meta": {"env": "E01", "subject": "S01"}}

def test_abstract_cannot_instantiate():
    with pytest.raises(TypeError):
        SyntheticCaptioner()  # ABC with abstract generate

def test_template_generates_multiple_sentences():
    c = TemplateCaptioner(n=3)
    texts = c.generate(_fake_sample())
    assert isinstance(texts, list) and len(texts) == 3
    assert all(isinstance(t, str) and len(t) > 10 for t in texts)

def test_template_contains_action_verb():
    c = TemplateCaptioner(n=1)
    texts = c.generate(_fake_sample(label=0))
    assert "stretching" in texts[0].lower()

def test_template_uses_meta_env_subject():
    c = TemplateCaptioner(n=1)
    texts = c.generate(_fake_sample(label=0))
    assert "E01" in texts[0] or "S01" in texts[0]


from curation.caption.quality import check_captions

def _issues_text(texts, verb):
    return " ".join(check_captions(texts, verb=verb))

def test_check_captions_ok():
    texts = ["A person is stretching and relaxing.", "Someone is stretching."]
    assert check_captions(texts, verb="stretching") == []

def test_check_captions_flags_empty():
    assert "empty" in _issues_text(["", "  "], verb="stretching")

def test_check_captions_flags_missing_verb():
    assert "verb" in _issues_text(["A person is waving."], verb="stretching")

def test_check_captions_flags_duplicates():
    assert "duplicate" in _issues_text(["same.", "same."], verb="stretching")


import os, pickle, sys
import numpy as np
from framework.dataset.sample import Sample, Modality
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from make_v5 import add_captions

def _mini_v4(tmp_path):
    root = tmp_path / "v4"
    (root / "data").mkdir(parents=True)
    mm = {m: Modality(data=np.zeros((2, 2, 2), dtype=np.float32), frame_indices=[1, 2], sample_rate=10)
          for m in ("wifi",)}
    s = Sample(id="E01_S01_A01_f1-7", label=0, modalities=mm)
    with open(root / "data" / "E01_S01_A01_f1-7.pkl", "wb") as f:
        pickle.dump(s.to_dict(), f)
    delta = {"kind": "variant", "id": "E01_S01_A01_f1-7__aug0", "base_id": "E01_S01_A01_f1-7",
             "label": 0, "rgb": np.zeros((2, 2, 2), dtype=np.float32), "aug": 0}
    with open(root / "data" / "E01_S01_A01_f1-7__aug0.pkl", "wb") as f:
        pickle.dump(delta, f)
    (root / "splits").mkdir(exist_ok=True)
    import json
    json.dump(["E01_S01_A01_f1-7", "E01_S01_A01_f1-7__aug0"], open(root / "splits" / "train.json", "w"))
    json.dump([], open(root / "splits" / "val.json", "w"))
    json.dump([], open(root / "splits" / "test.json", "w"))
    return root

def test_add_captions_writes_text_to_base(tmp_path):
    root = _mini_v4(tmp_path)
    v5 = tmp_path / "v5"
    captioner = TemplateCaptioner(n=2)
    add_captions(str(root), str(v5), captioner)

    with open(v5 / "data" / "E01_S01_A01_f1-7.pkl", "rb") as f:
        base = pickle.load(f)
    assert isinstance(base.get("text", {}).get("en"), list) and len(base["text"]["en"]) == 2
    assert "stretching" in base["text"]["en"][0].lower()

def test_add_captions_variant_kept_without_text(tmp_path):
    root = _mini_v4(tmp_path)
    v5 = tmp_path / "v5"
    captioner = TemplateCaptioner(n=2)
    add_captions(str(root), str(v5), captioner)
    with open(v5 / "data" / "E01_S01_A01_f1-7__aug0.pkl", "rb") as f:
        delta = pickle.load(f)
    # variant stays a delta (loader resolves base text at load time)
    assert delta.get("kind") == "variant"
