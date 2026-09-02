# tests/test_splits.py
import json
from framework.dataset.splits import split_annotations, _subject_of

def test_subject_of():
    assert _subject_of("./datasets/MMFi/E02/S19/A03") == "S19"

def test_split_annotations_disjoint_subjects(tmp_path):
    anns = []
    # 3 subjects x 2 actions
    for sub in ["S01", "S02", "S03"]:
        for act in ["A01", "A02"]:
            anns.append({"video_path": f"./datasets/MMFi/E01/{sub}/{act}",
                         "start_index": 0, "end_index": 9})
    train, val = split_annotations(anns, val_subjects=["S03"])
    train_subs = {_subject_of(a["video_path"]) for a in train}
    val_subs = {_subject_of(a["video_path"]) for a in val}
    assert val_subs == {"S03"}
    assert train_subs == {"S01", "S02"}
    assert train_subs.isdisjoint(val_subs)

def test_split_annotations_id_uses_frame_window(tmp_path):
    anns = [{"video_path": "./datasets/MMFi/E01/S01/A05", "start_index": 3, "end_index": 12}]
    train, _ = split_annotations(anns, val_subjects=[])
    assert train[0]["sample_id"] == "E01_S01_A05_f3-12"
