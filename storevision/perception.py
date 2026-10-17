"""Восприятие: видео → кадры → детекции людей (RF-DETR) → треки (Roboflow trackers).

Готовые компоненты:
  * RF-DETR (Apache-2.0) — детектор, COCO-класс person;
  * trackers (Apache-2.0) — BoT-SORT / ByteTrack / OC-SORT;
  * supervision (MIT) — Detections, аннотаторы.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import supervision as sv

from . import config as C


@dataclass
class CCTVEmulation:
    """Ухудшение стокового видео до уровня регистратора магазина."""
    width: int = 960          # типичный substream 960×540 / 704×576
    jpeg_quality: int = 35    # сжатие H.264 на низком битрейте ≈ артефакты JPEG q30-40
    grayscale: bool = False   # ночной ИК-режим

    def apply(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if w > self.width:
            frame = cv2.resize(frame, (self.width, int(h * self.width / w)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if self.grayscale:
            frame = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        return frame


def frames(cam: C.Camera, cctv: CCTVEmulation | None = None) -> Iterator[tuple[float, np.ndarray]]:
    """(время_с, кадр BGR) с частотой cam.process_fps."""
    cap = cv2.VideoCapture(cam.source)
    if not cap.isOpened():
        raise RuntimeError(f"не открыть источник {cam.source}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(1, round(src_fps / cam.process_fps))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            h, w = frame.shape[:2]
            if w > cam.resize_width:
                frame = cv2.resize(frame, (cam.resize_width, int(h * cam.resize_width / w)),
                                   interpolation=cv2.INTER_AREA)
            if cctv:
                frame = cctv.apply(frame)
            yield i / src_fps, frame
        i += 1
    cap.release()


_MODELS = {"rfdetr-nano": "RFDETRNano", "rfdetr-small": "RFDETRSmall", "rfdetr-medium": "RFDETRMedium",
           "rfdetr-base": "RFDETRBase", "rfdetr-large": "RFDETRLarge"}


class PersonDetector:
    def __init__(self, cfg: C.DetectorCfg):
        import rfdetr
        kwargs = {} if cfg.device == "auto" else {"device": cfg.device}
        self.model = getattr(rfdetr, _MODELS[cfg.model])(**kwargs)
        self.conf = cfg.confidence

    def __call__(self, frame_bgr: np.ndarray) -> sv.Detections:
        det = self.model.predict(frame_bgr[:, :, ::-1].copy(), threshold=self.conf)
        names = det.data.get("class_name")
        keep = (np.array(names) == "person") if names is not None else (det.class_id == 1)
        return det[keep]


class CameraDetector:
    """Детектор для конкретной камеры: общая модель + (опционально) SAHI-нарезка кадра."""

    def __init__(self, base: PersonDetector, slicing: C.SlicingCfg | None):
        self.base, self.slicing = base, slicing
        self._slicers: dict[tuple[int, int], sv.InferenceSlicer] = {}

    def _slicer(self, w: int, h: int) -> sv.InferenceSlicer:
        key = (w, h)
        if key not in self._slicers:
            gx, gy = self.slicing.grid
            ov = self.slicing.overlap
            sw, sh = int(np.ceil(w / gx * (1 + ov))), int(np.ceil(h / gy * (1 + ov)))
            self._slicers[key] = sv.InferenceSlicer(
                callback=self.base, slice_wh=(sw, sh),
                overlap_wh=(int(sw * ov), int(sh * ov)), iou_threshold=self.slicing.iou,
                overlap_metric=sv.OverlapMetric.IOS if self.slicing.metric == "ios" else sv.OverlapMetric.IOU)
        return self._slicers[key]

    def __call__(self, frame: np.ndarray) -> sv.Detections:
        if not self.slicing:
            return self.base(frame)
        h, w = frame.shape[:2]
        return self._slicer(w, h)(frame)


class CachedDetections:
    """Кэш детекций на диск: правила и зоны можно перенастраивать без повторного прогона сети."""

    def __init__(self, cam: C.Camera, dcfg: C.DetectorCfg, cache_dir: Path, tag: str = ""):
        sl = cam.slicing.model_dump_json() if cam.slicing else ""
        key = f"{cam.source}|{cam.process_fps}|{cam.resize_width}|{dcfg.model}|{dcfg.confidence}|{tag}|{sl}"
        self.path = cache_dir / f"det_{cam.id}_{hashlib.md5(key.encode()).hexdigest()[:10]}.npz"
        self.data: dict[str, np.ndarray] = {}
        if dcfg.cache and self.path.exists():
            self.data = dict(np.load(self.path))
        self.enabled = dcfg.cache
        self._detector: CameraDetector | None = None
        self._dcfg, self._slicing = dcfg, cam.slicing

    def get(self, idx: int, frame: np.ndarray) -> sv.Detections:
        k = str(idx)
        if k in self.data:
            arr = self.data[k]
        else:
            if self._detector is None:
                self._detector = CameraDetector(PersonDetector(self._dcfg), self._slicing)
            d = self._detector(frame)
            arr = np.hstack([d.xyxy, d.confidence[:, None]]) if len(d) else np.zeros((0, 5))
            self.data[k] = arr
        return sv.Detections(xyxy=arr[:, :4].astype(np.float32), confidence=arr[:, 4].astype(np.float32),
                             class_id=np.zeros(len(arr), dtype=int))

    def save(self):
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(self.path, **self.data)


def make_tracker(cfg: C.TrackerCfg, fps: float):
    import trackers
    buf = max(1, int(cfg.lost_track_buffer_s * fps))
    common = dict(lost_track_buffer=buf, frame_rate=fps, track_activation_threshold=0.5,
                  minimum_consecutive_frames=2)
    if cfg.type == "bytetrack":
        return trackers.ByteTrackTracker(**common)
    if cfg.type == "ocsort":
        return trackers.OCSORTTracker(**{k: v for k, v in common.items() if k != "track_activation_threshold"})
    return trackers.BoTSORTTracker(**common)


def uniform_ratio(frame: np.ndarray, box, ucfg: C.UniformCfg | None) -> float | None:
    """Доля пикселей торса в HSV-диапазонах формы. Торс = 20–55% высоты, центр 60% ширины."""
    if ucfg is None:
        return None
    x1, y1, x2, y2 = [int(v) for v in box]
    w, h = x2 - x1, y2 - y1
    if w < 12 or h < 30:
        return None
    crop = frame[max(0, y1 + int(.20 * h)):y1 + int(.55 * h), max(0, x1 + int(.2 * w)):x2 - int(.2 * w)]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for r in ucfg.hsv_ranges:
        mask |= cv2.inRange(hsv, np.array(r.lower), np.array(r.upper))
    return float(mask.mean() / 255)
