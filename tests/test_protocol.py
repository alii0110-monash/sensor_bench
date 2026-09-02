# tests/test_protocol.py
from framework.harness.protocol import build_protocol

def test_build_protocol_15_profiles():
    mods = ["wifi", "depth", "lidar", "mmwave"]
    p = build_protocol(mods, seeds=[0, 1, 2])
    profiles = p["profiles"]
    ids = [x["id"] for x in profiles]
    assert "full" in ids
    assert sum("miss-" in i for i in ids) == 4
    assert sum("only-" in i for i in ids) == 4
    assert sum("miss2-" in i for i in ids) == 6
    assert len(profiles) == 15
    assert p["seeds"] == [0, 1, 2]

def test_protocol_full_available_all():
    mods = ["wifi", "depth", "lidar", "mmwave"]
    p = build_protocol(mods, seeds=[0])
    full = [x for x in p["profiles"] if x["id"] == "full"][0]
    assert set(full["available"]) == set(mods)
