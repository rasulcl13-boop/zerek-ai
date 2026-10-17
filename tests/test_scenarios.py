"""Синтетические сценарии: проверяем логику правил без видео и нейросети.

Человек задаётся траекторией «зона → интервал времени», сотрудник — высокой долей
цвета формы (uniform_ratio=1), клиент — нулевой.
"""
from pathlib import Path

import pytest
from pydantic import ValidationError

from storevision import config as C
from storevision.engine import Engine, TrackIn

ROOT = Path(__file__).resolve().parent.parent


def center(cfg, zone_id):
    z = next(z for c in cfg.cameras for z in c.zones if z.id == zone_id)
    xs, ys = zip(*z.polygon)
    return sum(xs) / len(xs), sum(ys) / len(ys)


def box_at(cfg, cam, zone_id):
    """Рамка, у которой опорная точка камеры попадает в центр зоны."""
    x, y = center(cfg, zone_id)
    if cam.anchor == "center":
        return (x - .03, y - .08, x + .03, y + .08)
    return (x - .03, y - .16, x + .03, y)


def simulate(cfg, people, until, fps=5, camera=None):
    """people: список (track_id, is_staff, [(t0, t1, zone_id), ...])."""
    cam = next(c for c in cfg.cameras if camera in (None, c.id))
    eng = Engine(cfg)
    events = []
    n = int(until * fps)
    for i in range(n + 1):
        t = i / fps
        tracks = []
        for tid, staff, path in people:
            for t0, t1, zid in path:
                if t0 <= t < t1:
                    tracks.append(TrackIn(tid, box_at(cfg, cam, zid), 1.0 if staff else 0.0))
                    break
        events += eng.step(t, cam.id, tracks)
    return events


def alerts(ev, contains=""):
    return [e for e in ev if e.type == "alert" and contains in e.data["message"]]


def states(ev, obj):
    return [(round(e.t, 1), e.data["from"], e.data["to"], e.data["spent_s"]) for e in ev
            if e.type == "state" and e.data["object"] == obj]


# ---------- конфиги ----------

@pytest.mark.parametrize("name", sorted(p.name for p in (ROOT / "configs").glob("*.yaml")))
def test_all_configs_valid(name):
    C.load(ROOT / "configs" / name)


def test_bad_zone_reference_rejected():
    raw = C.load(ROOT / "configs/grocery_checkout.yaml").model_dump(by_alias=True)
    raw["objects"][0]["zones"]["wait"] = "nope"
    with pytest.raises(ValidationError, match="нет зоны 'nope'"):
        C.SiteConfig.model_validate(raw)


# ---------- продуктовый ----------

def test_grocery_queue_without_cashier_alerts_and_escalates():
    cfg = C.load(ROOT / "configs/grocery_checkout.yaml")
    ev = simulate(cfg, [
        (1, False, [(0, 30, "queue_1")]),                 # клиент стоит 30 с
        (2, True, [(15, 30, "cashier_1")]),               # кассир пришёл на 15-й секунде
    ], until=30)
    st = states(ev, "Касса 1")
    assert st[0][1:3] == ("idle", "unattended")
    assert st[1][1:3] == ("unattended", "serving")
    assert 13 <= st[1][3] <= 15   # 15 с до прихода − 2 с подтверждения + 0.4 с распознавания формы + 1 с перехода
    msgs = [e.data["message"] for e in alerts(ev, "Касса 1")]
    assert any("кассира нет" in m for m in msgs)
    assert any("менеджеру" in m for m in msgs)
    unattended = [e for e in ev if e.type == "state" and e.data["from"] == "unattended"]
    assert unattended[0].data["service_wait"] is True


def test_grocery_cashier_present_no_unattended():
    cfg = C.load(ROOT / "configs/grocery_checkout.yaml")
    ev = simulate(cfg, [
        (1, False, [(0, 20, "queue_1")]),
        (2, True, [(0, 20, "cashier_1")]),
    ], until=20)
    assert not alerts(ev, "кассира нет")
    assert ("idle", "serving") in [s[1:3] for s in states(ev, "Касса 1")]


def test_cashier_steps_away_briefly_is_not_a_signal():
    cfg = C.load(ROOT / "configs/grocery_checkout.yaml")
    ev = simulate(cfg, [
        (1, False, [(0, 20, "queue_1")]),
        (2, True, [(0, 8, "cashier_1"), (9.2, 20, "cashier_1")]),   # отошёл на 1.2 с (< for_s=2)
    ], until=20)
    assert not alerts(ev, "кассира нет")


def test_customer_in_staff_zone_not_promoted_quickly():
    cfg = C.load(ROOT / "configs/grocery_checkout.yaml")
    ev = simulate(cfg, [(1, False, [(0, 2, "cashier_1")])], until=3)
    assert not [e for e in ev if e.type == "role"]


# ---------- кафе ----------

def test_cafe_table_waits_for_waiter():
    cfg = C.load(ROOT / "configs/cafe_example.yaml")
    ev = simulate(cfg, [
        (1, False, [(0, 400, "table_1")]),               # гости сели и сидят
        (2, False, [(0, 400, "table_1")]),
        (9, True, [(230, 236, "table_1")]),              # официант подошёл на 230-й секунде на 6 с
    ], until=400, fps=2)
    st = states(ev, "Стол 1")
    assert st[0][1:3] == ("free", "waiting_first")
    assert st[1][1:3] == ("waiting_first", "served")
    assert 205 <= st[1][3] <= 216                         # 230 − 20 (посадка) + 4 (контакт)
    a = alerts(ev, "Стол 1")
    assert any("Айгерим" in e.data["message"] for e in a)
    assert all("менеджеру" not in e.data["message"] for e in a)   # 360 с не прошло


def test_cafe_waiter_passing_by_is_not_contact():
    cfg = C.load(ROOT / "configs/cafe_example.yaml")
    ev = simulate(cfg, [
        (1, False, [(0, 300, "table_2")]),
        (9, True, [(100, 101.5, "table_2")]),            # прошёл мимо за 1.5 с
    ], until=300, fps=2)
    assert [s[2] for s in states(ev, "Стол 2")] == ["waiting_first"]
    assert alerts(ev, "Стол 2")


def test_cafe_dirty_table_then_cleaned():
    cfg = C.load(ROOT / "configs/cafe_example.yaml")
    ev = simulate(cfg, [
        (1, False, [(0, 100, "table_1")]),
        (9, True, [(30, 36, "table_1"), (200, 207, "table_1")]),
    ], until=260, fps=2)
    assert [s[2] for s in states(ev, "Стол 1")] == ["waiting_first", "served", "dirty", "free"]


# ---------- одежда ----------

def test_fashion_customer_needs_help_then_helped():
    cfg = C.load(ROOT / "configs/fashion_example.yaml")
    ev = simulate(cfg, [
        (1, False, [(0, 200, "rack_women")]),
        (7, True, [(150, 160, "rack_women")]),
    ], until=200, fps=2)
    assert [s[2] for s in states(ev, "Стойка «Женское»")] == ["browsing", "needs_help", "helped"]
    a = [e.data["message"] for e in alerts(ev, "Стойка")]
    assert any("Мадина" in m for m in a)


def test_fashion_door_counts_customers_not_staff():
    cfg = C.load(ROOT / "configs/fashion_example.yaml")
    eng = Engine(cfg)
    cust = [.99, .99, .99, .97, .93, .90]    # клиент заходит (снизу вверх через линию y=.95)
    staff = [.90, .90, .90, .93, .97, .99]   # сотрудник выходит
    for i, (yc, ys) in enumerate(zip(cust, staff)):
        eng.step(i * .2, "floor", [TrackIn(1, (.28, yc - .2, .32, yc), 0.0),
                                   TrackIn(2, (.38, ys - .2, .42, ys), 1.0)])
    assert eng.counters["door"] == {"in": 1, "out": 0}
