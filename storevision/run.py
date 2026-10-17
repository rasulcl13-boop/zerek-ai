"""Прогон площадки по конфигу.

    python -m storevision.run configs/grocery_checkout.yaml [--cctv] [--no-video]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from . import config as C
from .engine import Engine, TrackIn
from .overlay import Overlay
from .perception import CachedDetections, CCTVEmulation, frames, make_tracker, uniform_ratio
from .report import build_report
from .sinks import EventStore, Notifier

ROOT = Path(__file__).resolve().parent.parent


def run(cfg_path: str, cctv: bool = False, video: bool = True, quiet: bool = False) -> dict:
    return run_cfg(C.load(cfg_path), cctv=cctv, video=video, quiet=quiet)


def run_cfg(cfg: C.SiteConfig, cctv: bool = False, video: bool = True, quiet: bool = False,
            out: Path | None = None) -> dict:
    out = out or ROOT / cfg.output.dir / cfg.site.id / ("cctv" if cctv else "raw")
    out.mkdir(parents=True, exist_ok=True)
    (out / "alerts").mkdir(exist_ok=True)
    store = EventStore(out / "events.db", cfg.site.id)
    notify = Notifier(cfg.output.telegram)
    engine = Engine(cfg)
    emu = CCTVEmulation() if cctv else None
    stats = {"frames": 0, "alerts": 0, "sec": 0.0}
    t_start = time.time()

    for cam in cfg.cameras:  # MVP: камеры по очереди; в проде — поток на камеру
        cache = CachedDetections(cam, cfg.detector, ROOT / "out" / "cache", tag="cctv" if cctv else "")
        tracker = make_tracker(cfg.tracker, cam.process_fps)
        overlay = Overlay(cam, cfg)
        writer, tmp = None, out / f"{cam.id}.tmp.mp4"
        for idx, (t, frame) in enumerate(frames(cam, emu)):
            h, w = frame.shape[:2]
            det = cache.get(idx, frame)
            tracked = tracker.update(det, frame=frame)
            ids = tracked.tracker_id if tracked.tracker_id is not None else np.array([])
            keep = ids >= 0 if len(ids) else np.zeros(0, bool)
            boxes, ids = tracked.xyxy[keep], ids[keep]
            tracks = [TrackIn(int(i), (b[0] / w, b[1] / h, b[2] / w, b[3] / h),
                              uniform_ratio(frame, b, cfg.staff.uniform))
                      for b, i in zip(boxes, ids)]
            events = engine.step(t, cam.id, tracks)
            store.write(events)

            img = overlay.draw(frame, t, engine, boxes, ids) if video else None
            for e in events:
                if e.type == "alert":
                    stats["alerts"] += 1
                    overlay.push_alert(t, e.data["severity"], e.data["message"])
                    snap = out / "alerts" / f"{cam.id}_{t:07.1f}_{e.data['rule'].replace(':', '-')}.jpg"
                    if img is not None:
                        img = overlay.draw(frame, t, engine, boxes, ids)
                        cv2.imwrite(str(snap), img)
                    if quiet:
                        e.data["notify"] = [n for n in e.data["notify"] if n != "console"]
                    notify(e, snap)
            if video:
                if writer is None:
                    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), cam.process_fps, (w, h))
                writer.write(img)
            stats["frames"] += 1
            stats["sec"] = max(stats["sec"], t)
        store.write(engine.flush(cam.id))
        cache.save()
        if writer:
            writer.release()
            final = out / f"{cam.id}.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            "-crf", "26", "-movflags", "+faststart", str(final)], check=True)
            tmp.unlink()

    store.close()
    stats["wall_s"] = round(time.time() - t_start, 1)
    report = build_report(out / "events.db", store.run, cfg)
    report["perf"] = stats
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--cctv", action="store_true", help="эмулировать качество регистратора")
    ap.add_argument("--no-video", action="store_true")
    a = ap.parse_args()
    rep = run(a.config, cctv=a.cctv, video=not a.no_video)
    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
