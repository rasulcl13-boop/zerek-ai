"""Оверлей для демо-видео: зоны, роли, линии, состояния объектов, оповещения."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import config as C
from .engine import Engine

FONT_PATHS = ["/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
              "/System/Library/Fonts/Supplemental/Arial.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
ZONE_COLORS = {"staff_area": (255, 140, 0), "queue": (0, 165, 255), "table": (180, 105, 255),
               "fitting_room": (180, 105, 255), "service": (255, 140, 0), "shelf": (0, 200, 200)}
STAFF, CUSTOMER = (60, 200, 60), (240, 240, 240)
STATE_COLORS = {"warning": (0, 165, 255), "critical": (0, 0, 255)}


def _font(size):
    for p in FONT_PATHS:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


class Overlay:
    def __init__(self, cam: C.Camera, cfg: C.SiteConfig):
        self.cam, self.cfg = cam, cfg
        k = max(0.45, cam.resize_width / 1280)          # шрифт под ширину кадра
        self.f_s, self.f_m, self.f_l = _font(int(15 * k)), _font(int(18 * k)), _font(int(22 * k))
        self.k = k
        self.alerts: list[tuple[float, str, str]] = []

    def push_alert(self, t, severity, message):
        self.alerts.append((t, severity, message))

    def draw(self, frame: np.ndarray, t: float, engine: Engine, boxes, ids) -> np.ndarray:
        h, w = frame.shape[:2]
        P = lambda p: (int(p[0] * w), int(p[1] * h))
        img = frame.copy()
        layer = img.copy()
        snap = engine.snapshot(t)
        for z in self.cam.zones:
            pts = np.array([P(p) for p in z.polygon], dtype=np.int32)
            col = ZONE_COLORS.get(z.kind, (200, 200, 200))
            cv2.fillPoly(layer, [pts], col)
            cv2.polylines(img, [pts], True, col, 2)
        img = cv2.addWeighted(layer, 0.18, img, 0.82, 0)
        for l in self.cam.lines:
            cv2.line(img, P(l.start), P(l.end), (255, 0, 255), 3)

        texts = []  # (xy, text, color, font)
        cam_tracks = engine.tracks[self.cam.id]
        for box, tid in zip(boxes, ids):
            st = cam_tracks.get(int(tid))
            role = st.role if st else "customer"
            col = STAFF if role == "staff" else CUSTOMER
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
            dwell = ""
            if st and st.zones:
                zid, t0 = min(st.zones.items(), key=lambda kv: kv[1])
                dwell = f" {t - t0:.0f}с"
            texts.append(((x1, max(0, y1 - 20)), f"#{tid} {'сотр' if role == 'staff' else 'клиент'}{dwell}",
                          col, self.f_s))

        for z in self.cam.zones:
            s = snap["zones"][z.id]
            x, y = P(z.polygon[0])
            texts.append(((x + 4, y + 4), f"{z.label or z.id}: кл {s['customer']} / сотр {s['staff']}",
                          ZONE_COLORS.get(z.kind, (220, 220, 220)), self.f_s))
        for l in self.cam.lines:
            c = snap["lines"][l.id]
            ex, ey = P(l.end)
            texts.append(((min(ex, w - int(260 * self.k)), ey + 4), f"{l.label or l.id}: вход {c['in']} / выход {c['out']}", (255, 0, 255), self.f_m))

        # состояния объектов — плашка справа сверху
        y = 10
        for o in engine.objects or []:
            if o.obj.camera != self.cam.id:
                continue
            sdef = o.m.states[o.state]
            over = [a for a in sdef.alerts if t - o.entered >= a.after_s]
            col = STATE_COLORS.get(over[-1].severity, (255, 255, 255)) if over else (255, 255, 255)
            owner = f" ({o.obj.owner})" if o.obj.owner else ""
            texts.append(((w - int(430 * self.k), y), f"{o.obj.id}{owner}: {sdef.label or o.state} · {t - o.entered:.0f}с",
                          col, self.f_m))
            y += int(26 * self.k)

        # оповещения — снизу, 4 секунды
        self.alerts = [a for a in self.alerts if t - a[0] < 4]
        for i, (_, sev, msg) in enumerate(reversed(self.alerts[-3:])):
            yy = h - int(40 * self.k) - i * int(34 * self.k)
            cv2.rectangle(img, (0, yy - 4), (w, yy + int(28 * self.k)), STATE_COLORS.get(sev, (80, 80, 80)), -1)
            texts.append(((12, yy), msg, (255, 255, 255), self.f_l))

        texts.append(((10, 10), f"{self.cfg.site.name} · {self.cam.id} · t={t:5.1f}с", (255, 255, 255), self.f_m))
        return self._text(img, texts)

    @staticmethod
    def _text(img, texts):
        pil = Image.fromarray(img[:, :, ::-1])
        d = ImageDraw.Draw(pil)
        for (x, y), s, col, font in texts:
            rgb = (col[2], col[1], col[0])
            d.text((x, y), s, font=font, fill=rgb, stroke_width=2, stroke_fill=(0, 0, 0))
        return np.asarray(pil)[:, :, ::-1].copy()
