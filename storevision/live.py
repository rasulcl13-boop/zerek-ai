"""Живой режим: RTSP (через Frigate/go2rtc) → анализ → SQLite + оповещения + дашборд.

    python -m storevision.live configs/live_demo.yaml --port 8088
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import sqlite3
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;5000000")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from . import config as C  # noqa: E402
from .engine import Engine, Event, TrackIn  # noqa: E402
from .overlay import Overlay  # noqa: E402
from .perception import CameraDetector, PersonDetector, make_tracker, uniform_ratio  # noqa: E402
from .sinks import SCHEMA, Notifier  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class CamStatus:
    id: str
    source: str
    state: str = "starting"          # starting / online / reconnecting
    fps: float = 0.0
    last_frame: float = 0.0
    reconnects: int = 0
    error: str = ""
    infer_ms: float = 0.0


class StoreWriter(threading.Thread):
    """Единственный писатель SQLite; дашборд читает своими соединениями (WAL)."""

    def __init__(self, path: Path, site: str):
        super().__init__(daemon=True, name="store")
        self.path, self.site = path, site
        self.q: queue.Queue[list[Event]] = queue.Queue()
        self.run_id = time.strftime("live-%Y%m%d-%H%M%S")

    def run(self):
        db = sqlite3.connect(self.path)
        db.execute("PRAGMA journal_mode=WAL")
        db.executescript(SCHEMA)
        while True:
            batch = self.q.get()
            try:
                while True:
                    batch += self.q.get_nowait()
            except queue.Empty:
                pass
            db.executemany("INSERT INTO events(run,site,camera,t,type,data) VALUES(?,?,?,?,?,?)",
                           [(self.run_id, self.site, e.camera, e.t, e.type, json.dumps(e.data, ensure_ascii=False))
                            for e in batch])
            db.commit()


class LiveSystem:
    def __init__(self, cfg: C.SiteConfig, out: Path):
        self.cfg, self.out = cfg, out
        (out / "alerts").mkdir(parents=True, exist_ok=True)
        self.db_path = out / "events.db"
        self.engine = Engine(cfg)
        self.lock = threading.Lock()            # engine — общий для камер
        self.det_lock = threading.Lock()        # одна модель на всех, инференс по очереди
        base = PersonDetector(cfg.detector)
        self.detectors = {c.id: CameraDetector(base, c.slicing) for c in cfg.cameras}
        self.notify = Notifier(cfg.output.telegram)
        self.store = StoreWriter(self.db_path, cfg.site.id)
        self.status = {c.id: CamStatus(c.id, c.source) for c in cfg.cameras}
        self.frames: dict[str, bytes] = {}
        self.started = time.time()

    def start(self):
        self.store.start()
        for cam in self.cfg.cameras:
            threading.Thread(target=self._camera_loop, args=(cam,), daemon=True, name=f"cam-{cam.id}").start()

    # ---------- камера ----------

    def _reader(self, cam: C.Camera, cap, latest: dict, stop: threading.Event):
        """Читаем поток без остановки и держим только последний кадр — нет накопления задержки."""
        while not stop.is_set():
            ok, frame = cap.read()
            if not ok:
                latest["eof"] = True
                return
            latest["frame"], latest["t"] = frame, time.time()

    def _camera_loop(self, cam: C.Camera):
        st = self.status[cam.id]
        backoff = 1.0
        while True:
            cap = cv2.VideoCapture(cam.source, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                st.state, st.error = "reconnecting", "нет подключения к потоку"
                st.reconnects += 1
                time.sleep(backoff); backoff = min(backoff * 2, 30)
                continue
            st.state, st.error, backoff = "online", "", 1.0
            latest: dict = {}
            stop = threading.Event()
            threading.Thread(target=self._reader, args=(cam, cap, latest, stop), daemon=True).start()
            self._process(cam, latest, st)
            stop.set(); cap.release()
            st.state, st.error = "reconnecting", "поток прервался"
            st.reconnects += 1
            time.sleep(1)

    def _process(self, cam: C.Camera, latest: dict, st: CamStatus):
        tracker = make_tracker(self.cfg.tracker, cam.process_fps)
        overlay = Overlay(cam, self.cfg)
        period = 1.0 / cam.process_fps
        last_seen_t, ticks = 0.0, []
        while not latest.get("eof"):
            t0 = time.time()
            frame, ft = latest.get("frame"), latest.get("t", 0.0)
            if frame is None or ft == last_seen_t:
                if t0 - (ft or self.started) > 10:      # 10 с без кадров — переподключаемся
                    return
                time.sleep(0.02); continue
            last_seen_t = ft
            h, w = frame.shape[:2]
            if w > cam.resize_width:
                frame = cv2.resize(frame, (cam.resize_width, int(h * cam.resize_width / w)),
                                   interpolation=cv2.INTER_AREA)
                h, w = frame.shape[:2]
            with self.det_lock:                       # инференс по очереди; время — без ожидания очереди
                ti = time.time()
                det = self.detectors[cam.id](frame)
                st.infer_ms = (time.time() - ti) * 1000
            tracked = tracker.update(det, frame=frame)
            ids = tracked.tracker_id if tracked.tracker_id is not None else np.array([], dtype=int)
            keep = ids >= 0 if len(ids) else np.zeros(0, bool)
            boxes, ids = tracked.xyxy[keep], ids[keep]
            tracks = [TrackIn(int(i), (b[0] / w, b[1] / h, b[2] / w, b[3] / h),
                              uniform_ratio(frame, b, self.cfg.staff.uniform)) for b, i in zip(boxes, ids)]
            with self.lock:
                events = self.engine.step(ft, cam.id, tracks)
                img = overlay.draw(frame, ft, self.engine, boxes, ids)
                for e in events:
                    if e.type == "alert":
                        overlay.push_alert(ft, e.data["severity"], e.data["message"])
                        img = overlay.draw(frame, ft, self.engine, boxes, ids)
                        self._on_alert(cam, e, img)
            self.store.q.put(events)
            ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                self.frames[cam.id] = jpg.tobytes()
            st.last_frame = ft
            ticks = [x for x in ticks if t0 - x < 5] + [t0]
            st.fps = round(len(ticks) / 5, 1) if t0 - self.started > 5 else 0
            time.sleep(max(0.0, period - (time.time() - t0)))

    def _on_alert(self, cam: C.Camera, e: Event, img):
        name = f"{cam.id}_{int(e.t)}_{e.data['rule'].replace(':', '-').replace(' ', '_')}.jpg"
        cv2.imwrite(str(self.out / "alerts" / name), img)
        e.data["snapshot"] = name
        fr = self.cfg.frigate
        if fr and cam.frigate_camera:
            e.data["clip"] = (f"{fr.url}/api/{cam.frigate_camera}/start/{int(e.t - fr.clip_before_s)}"
                              f"/end/{int(e.t + fr.clip_after_s)}/clip.mp4")
        self.notify(e, self.out / "alerts" / name)

    # ---------- для дашборда ----------

    def live_state(self) -> dict:
        now = time.time()
        with self.lock:
            snap = self.engine.snapshot(now)
            people = {c.id: {"customer": sum(1 for s in self.engine.tracks[c.id].values() if s.role == "customer"),
                             "staff": sum(1 for s in self.engine.tracks[c.id].values() if s.role == "staff")}
                      for c in self.cfg.cameras}
        zones = []
        for c in self.cfg.cameras:
            for z in c.zones:
                zones.append({"id": z.id, "label": z.label or z.id, "kind": z.kind, "camera": c.id,
                              **snap["zones"][z.id]})
        objects = []
        for o in self.engine.objects or []:
            sdef = o.m.states[o.state]
            fired = [a for a in sdef.alerts if now - o.entered >= a.after_s]
            objects.append({"id": o.obj.id, "owner": o.obj.owner, "camera": o.obj.camera, "state": o.state,
                            "label": sdef.label or o.state, "for_s": round(now - o.entered, 1),
                            "severity": fired[-1].severity if fired else ("ok" if not sdef.counts_as_service_wait
                                                                          else "pending")})
        cams = []
        for c in self.cfg.cameras:
            s = self.status[c.id]
            cams.append({"id": c.id, "state": s.state if now - s.last_frame < 10 or s.state != "online"
                         else "stalled", "fps": s.fps, "infer_ms": round(s.infer_ms),
                         "reconnects": s.reconnects, "error": s.error, "frigate": c.frigate_camera,
                         "slicing": f"SAHI {c.slicing.grid[0]}×{c.slicing.grid[1]}" if c.slicing else None,
                         "people": people[c.id]})
        return {"site": self.cfg.site.model_dump(), "now": now, "uptime_s": round(now - self.started),
                "cameras": cams, "zones": zones, "objects": objects,
                "lines": snap["lines"], "frigate_ui": self.cfg.frigate.url if self.cfg.frigate else None}


def check_frigate(url: str) -> str:
    try:
        with urllib.request.urlopen(f"{url}/api/version", timeout=3) as r:
            return r.read().decode().strip()
    except Exception as ex:
        return f"недоступен ({ex})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8088)
    a = ap.parse_args()
    cfg = C.load(a.config)
    out = ROOT / cfg.output.dir / cfg.site.id / "live"
    if cfg.frigate:
        print(f"Frigate {cfg.frigate.url}: {check_frigate(cfg.frigate.url)}")
    system = LiveSystem(cfg, out)
    system.start()
    import uvicorn
    from .dashboard import create_app
    print(f"Дашборд: http://{a.host}:{a.port}")
    uvicorn.run(create_app(system), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
