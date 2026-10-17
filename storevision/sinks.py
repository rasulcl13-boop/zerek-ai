"""Куда уходят события: SQLite-журнал, консоль, Telegram."""
from __future__ import annotations

import json
import os
import sqlite3
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

from . import config as C
from .engine import Event

SCHEMA = """
CREATE TABLE IF NOT EXISTS events(
  id INTEGER PRIMARY KEY, run TEXT, site TEXT, camera TEXT, t REAL, type TEXT, data TEXT);
CREATE INDEX IF NOT EXISTS ev_type ON events(run, type);
"""


class EventStore:
    """Храним события, а не видео: метрики любой отрасли считаются из этой таблицы."""

    def __init__(self, path: Path, site: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.executescript(SCHEMA)
        self.site, self.run = site, uuid.uuid4().hex[:8]

    def write(self, events: list[Event]):
        self.db.executemany(
            "INSERT INTO events(run,site,camera,t,type,data) VALUES(?,?,?,?,?,?)",
            [(self.run, self.site, e.camera, e.t, e.type, json.dumps(e.data, ensure_ascii=False))
             for e in events])

    def close(self):
        self.db.commit(); self.db.close()


class Notifier:
    def __init__(self, cfg: C.TelegramCfg):
        self.cfg = cfg
        self.token = os.environ.get(cfg.token_env)
        self.chat = os.environ.get(cfg.chat_id_env)
        self.tg = cfg.enabled and bool(self.token and self.chat)

    def __call__(self, e: Event, snapshot: Path | None = None):
        d = e.data
        icon = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}[d["severity"]]
        text = f"{icon} [{e.camera} {e.t:6.1f}с] {d['message']}"
        if "console" in d["notify"]:
            print(text)
        if "telegram" in d["notify"] and self.tg:
            self._send(text, snapshot if self.cfg.send_snapshot else None)

    def _send(self, text: str, photo: Path | None):
        base = f"https://api.telegram.org/bot{self.token}"
        try:
            if photo and photo.exists():
                boundary = uuid.uuid4().hex
                body = b"".join([
                    f"--{boundary}\r\nContent-Disposition: form-data; name=\"chat_id\"\r\n\r\n{self.chat}\r\n".encode(),
                    f"--{boundary}\r\nContent-Disposition: form-data; name=\"caption\"\r\n\r\n{text}\r\n".encode(),
                    f"--{boundary}\r\nContent-Disposition: form-data; name=\"photo\"; filename=\"a.jpg\"\r\n"
                    f"Content-Type: image/jpeg\r\n\r\n".encode(), photo.read_bytes(), f"\r\n--{boundary}--\r\n".encode()])
                req = urllib.request.Request(f"{base}/sendPhoto", data=body,
                                             headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
            else:
                req = urllib.request.Request(f"{base}/sendMessage",
                                             data=urllib.parse.urlencode({"chat_id": self.chat, "text": text}).encode())
            urllib.request.urlopen(req, timeout=10).read()
        except Exception as ex:  # оповещение не должно ронять конвейер
            print(f"  (telegram не отправлен: {ex})")
