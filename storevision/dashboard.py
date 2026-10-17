"""Дашборд: FastAPI + одна HTML-страница. Данные — живое состояние движка и журнал событий."""
from __future__ import annotations

import json
import sqlite3
import statistics
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse

WEB = Path(__file__).resolve().parent / "web"

RANGES = {  # окно → (начало, шаг корзины, формат подписи)
    "1h": (lambda now, tz: now - timedelta(hours=1), timedelta(minutes=5), "%H:%M"),
    "today": (lambda now, tz: now.replace(hour=0, minute=0, second=0, microsecond=0), timedelta(hours=1), "%H:00"),
    "7d": (lambda now, tz: (now - timedelta(days=6)).replace(hour=0, minute=0, second=0, microsecond=0),
           timedelta(days=1), "%d.%m"),
}


def _pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def stats(db_path: Path, tzname: str, rng: str, zone_labels: dict, line_labels: dict) -> dict:
    tz = ZoneInfo(tzname)
    now = datetime.now(tz)
    start_fn, step, fmt = RANGES[rng]
    start = start_fn(now, tz)
    t0 = start.timestamp()
    buckets = []
    b = start
    while b <= now:
        buckets.append({"label": b.strftime(fmt), "t": b.timestamp(), "in": 0, "out": 0})
        b += step
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    rows = db.execute("SELECT camera,t,type,data FROM events WHERE t>=? AND type IN "
                      "('line_cross','zone_exit','state','alert') ORDER BY t", (t0,)).fetchall()
    db.close()

    dwell = defaultdict(list)
    waits = defaultdict(list)
    served = defaultdict(int)
    alerts = {"info": 0, "warning": 0, "critical": 0}
    by_line = defaultdict(lambda: {"in": 0, "out": 0})
    for cam, t, typ, d in rows:
        d = json.loads(d)
        if typ == "line_cross" and d["role"] == "customer":
            i = min(len(buckets) - 1, int((t - t0) // step.total_seconds()))
            buckets[i][d["dir"]] += 1
            by_line[d["line"]][d["dir"]] += 1
        elif typ == "zone_exit" and d["role"] == "customer" and d["dwell_s"] >= 2:
            dwell[d["zone"]].append(d["dwell_s"])
        elif typ == "state":
            if d.get("service_wait"):
                waits[d.get("owner") or "—"].append(d["spent_s"])
            if d["to"] in ("serving", "served", "helped"):
                served[d.get("owner") or "—"] += 1
        elif typ == "alert":
            alerts[d["severity"]] = alerts.get(d["severity"], 0) + 1

    owners = sorted(set(waits) | set(served))
    return {
        "range": rng, "from": t0, "now": now.timestamp(),
        "traffic": buckets,
        "traffic_total": {"in": sum(x["in"] for x in buckets), "out": sum(x["out"] for x in buckets)},
        "lines": [{"id": k, "label": line_labels.get(k, k), **v} for k, v in by_line.items()],
        "dwell": [{"zone": z, "label": zone_labels.get(z, z), "visits": len(v),
                   "median_s": round(statistics.median(v), 1), "p90_s": round(_pct(v, .9), 1)}
                  for z, v in sorted(dwell.items(), key=lambda kv: -len(kv[1]))],
        "service": [{"owner": o, "served": served.get(o, 0), "waits": len(waits.get(o, [])),
                     "median_wait_s": round(statistics.median(waits[o]), 1) if waits.get(o) else None,
                     "p90_wait_s": round(_pct(waits[o], .9), 1) if waits.get(o) else None,
                     "max_wait_s": round(max(waits[o]), 1) if waits.get(o) else None}
                    for o in owners],
        "alerts": alerts,
    }


def recent_alerts(db_path: Path, limit: int) -> list[dict]:
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    rows = db.execute("SELECT camera,t,data FROM events WHERE type='alert' ORDER BY t DESC LIMIT ?",
                      (limit,)).fetchall()
    db.close()
    return [{"camera": c, "t": t, **json.loads(d)} for c, t, d in rows]


def create_app(system) -> FastAPI:
    app = FastAPI(title="store-vision")
    cfg = system.cfg
    zone_labels = {z.id: z.label or z.id for c in cfg.cameras for z in c.zones}
    line_labels = {l.id: l.label or l.id for c in cfg.cameras for l in c.lines}

    def wait_db():
        for _ in range(50):
            if system.db_path.exists():
                return
            time.sleep(0.1)
        raise HTTPException(503, "журнал ещё не создан")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (WEB / "index.html").read_text(encoding="utf-8")

    @app.get("/api/state")
    def state():
        return system.live_state()

    @app.get("/api/stats")
    def get_stats(range: str = "today"):
        if range not in RANGES:
            raise HTTPException(400, "range: 1h | today | 7d")
        wait_db()
        return stats(system.db_path, cfg.site.timezone, range, zone_labels, line_labels)

    @app.get("/api/alerts")
    def alerts(limit: int = 30):
        wait_db()
        return recent_alerts(system.db_path, min(limit, 200))

    @app.get("/api/frame/{cam}.jpg")
    def frame(cam: str):
        jpg = system.frames.get(cam)
        if not jpg:
            raise HTTPException(404, "кадра ещё нет")
        return Response(jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/stream/{cam}")
    def stream(cam: str):
        if cam not in system.status:
            raise HTTPException(404)

        def gen():
            last = None
            while True:
                jpg = system.frames.get(cam)
                if jpg is not None and jpg is not last:
                    last = jpg
                    yield b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                time.sleep(0.05)
        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=f")

    @app.get("/alerts/{name}")
    def snapshot(name: str):
        p = (system.out / "alerts" / name).resolve()
        if p.parent != (system.out / "alerts").resolve() or not p.exists():
            raise HTTPException(404)
        return FileResponse(p)

    return app
