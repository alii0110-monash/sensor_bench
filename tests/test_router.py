# tests/test_router.py
from framework.models.router import TokenRouter

def test_route_missing_modality_zero():
    r = TokenRouter(k_max=8)
    avail = {"wifi": True, "depth": False, "lidar": True, "mmwave": True, "rgb": True}
    budget = 32
    counts = r.route(avail, budget)
    assert counts["depth"] == 0
    assert counts["wifi"] >= 1

def test_route_respects_budget():
    r = TokenRouter(k_max=8)
    avail = {m: True for m in ["wifi", "depth", "lidar", "mmwave", "rgb"]}
    counts = r.route(avail, budget=10)
    assert sum(counts.values()) <= 10
    assert all(1 <= v <= 8 for v in counts.values())

def test_route_extreme_budget_fallback():
    r = TokenRouter(k_max=8)
    avail = {m: True for m in ["wifi", "depth", "lidar", "mmwave", "rgb"]}
    counts = r.route(avail, budget=0)
    assert all(v == 0 for v in counts.values())  # 回退纯文本

def test_route_prefix_stable():
    # 截取稳定性: 预算从 40 → 15 → 5 单调不减
    r = TokenRouter(k_max=8)
    avail = {m: True for m in ["wifi", "depth", "lidar", "mmwave", "rgb"]}
    c40 = r.route(avail, 40); c15 = r.route(avail, 15); c5 = r.route(avail, 5)
    s40 = sum(c40.values()); s15 = sum(c15.values()); s5 = sum(c5.values())
    assert s40 >= s15 >= s5
