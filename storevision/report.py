"""Отчёт из журнала событий: трафик, пребывание в зонах, ожидание обслуживания, оповещения."""
from __future__ import annotations

import json
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path

from . import config as C


def _pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(q * len(xs)))], 1)


def build_report(db_path: Path, run: str, cfg: C.SiteConfig) -> dict:
    db = sqlite3.connect(db_path)
    rows = [(cam, t, typ, json.loads(d)) for cam, t, typ, d in
            db.execute("SELECT camera,t,type,data FROM events WHERE run=? ORDER BY t", (run,))]
    db.close()

    traffic = defaultdict(lambda: {"in": 0, "out": 0, "staff_crossings": 0})
    dwell = defaultdict(list)
    roles = {}
    alerts, waits = [], defaultdict(list)
    for cam, t, typ, d in rows:
        if typ == "line_cross":
            if d["role"] == "customer":
                traffic[d["line"]][d["dir"]] += 1
            else:
                traffic[d["line"]]["staff_crossings"] += 1
        elif typ == "zone_exit" and d["role"] == "customer" and d["dwell_s"] >= 1.0:
            dwell[d["zone"]].append(d["dwell_s"])
        elif typ == "track_end":
            roles[(cam, d["track"])] = (d["role"], d["life_s"])
        elif typ == "alert":
            alerts.append({"t": round(t, 1), "camera": cam, **{k: d[k] for k in ("rule", "severity", "message")}})
        elif typ == "state" and d.get("service_wait"):
            waits[d.get("owner") or "—"].append(d["spent_s"])

    long_tracks = [r for r in roles.values() if r[1] >= 1.0]
    return {
        "site": cfg.site.model_dump(),
        "people": {"tracks_≥1s": len(long_tracks),
                   "staff": sum(1 for r in long_tracks if r[0] == "staff"),
                   "customers": sum(1 for r in long_tracks if r[0] == "customer")},
        "traffic": dict(traffic),
        "zone_dwell_customers": {z: {"visits": len(v), "median_s": round(statistics.median(v), 1),
                                     "p90_s": _pct(v, .9), "max_s": round(max(v), 1)}
                                 for z, v in dwell.items()},
        "service_wait_by_owner": {o: {"cases": len(v), "median_s": round(statistics.median(v), 1),
                                      "p90_s": _pct(v, .9)} for o, v in waits.items()},
        "alerts": alerts,
    }
