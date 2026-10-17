"""Схема конфига площадки.

Один YAML = одна площадка (магазин, кафе, бутик). Код ядра общий, отрасль
отличается только конфигом: зонами, линиями, объектами с состояниями и правилами.
Координаты зон и линий нормированы (0..1) — конфиг не зависит от разрешения камеры.
"""
from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

Point = tuple[float, float]
Who = Literal["customer", "staff", "any"]
Op = Literal[">", ">=", "<", "<=", "==", "!="]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------- геометрия ----------

class Zone(Strict):
    id: str
    kind: Literal["staff_area", "queue", "service", "table", "fitting_room",
                  "shelf", "entrance", "storefront", "generic"] = "generic"
    polygon: list[Point] = Field(min_length=3)
    label: str | None = None


class Line(Strict):
    id: str
    start: Point
    end: Point
    # Направление «вход» — то, которое supervision считает `in` (слева→направо
    # относительно вектора start→end). Если на месте окажется наоборот — swap.
    swap_direction: bool = False
    # Гистерезис: пересечение засчитывается, когда точка ушла за линию дальше чем на
    # hysteresis (доля кадра). Гасит двойной счёт, когда человек топчется на пороге.
    hysteresis: float = 0.02
    label: str | None = None


class SlicingCfg(Strict):
    """SAHI-нарезка кадра (supervision.InferenceSlicer): мелкие люди вдали становятся крупнее для
    детектора. На Mall: ошибка подсчёта 4,6 → 3,1 чел. Цена — ~grid[0]×grid[1] инференсов на кадр."""
    grid: tuple[int, int] = (2, 2)     # плиток по ширине × высоте
    overlap: float = 0.2               # перекрытие соседних плиток, доля плитки
    iou: float = 0.5                   # порог подавления дублей при склейке плиток
    # ios — пересечение к меньшей рамке: схлопывает обрезанного на стыке плиток человека с целым
    # (с iou такие пары не подавляются → трекер видит «призрака», лишние проходы через дверь)
    metric: Literal["ios", "iou"] = "iou"     # на CAVIAR ios хуже (77 % ошибок против 19 %)


class Camera(Strict):
    id: str
    source: str                      # путь к файлу или rtsp://
    process_fps: float = 10          # сколько кадров в секунду анализировать
    resize_width: int = 1280
    # Точка, по которой человек «стоит» в зоне. bottom — ноги (камера сверху, как CCTV);
    # center — центр рамки (камера на уровне глаз, ноги закрыты кассой/тележкой).
    anchor: Literal["bottom", "center"] = "bottom"
    zones: list[Zone] = []
    lines: list[Line] = []
    frigate_camera: str | None = None   # имя камеры во Frigate → ссылки на клипы записи
    slicing: SlicingCfg | None = None   # включать на камерах с толпой/мелкими людьми (вход, кассы)


# ---------- восприятие ----------

class DetectorCfg(Strict):
    model: Literal["rfdetr-nano", "rfdetr-small", "rfdetr-medium", "rfdetr-base",
                   "rfdetr-large"] = "rfdetr-medium"
    confidence: float = 0.4
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    cache: bool = True               # кэш детекций → быстрый пересчёт правил


class TrackerCfg(Strict):
    type: Literal["bytetrack", "botsort", "ocsort"] = "botsort"
    lost_track_buffer_s: float = 2.0


class HSVRange(Strict):
    lower: tuple[int, int, int]
    upper: tuple[int, int, int]


class UniformCfg(Strict):
    """Форма сотрудников: доля пикселей торса в заданных HSV-диапазонах."""
    hsv_ranges: list[HSVRange]
    min_ratio: float = 0.25
    min_samples: int = 3


class StaffZoneCfg(Strict):
    """Кто провёл в staff-зоне ≥ min_s — сотрудник (за кассой клиенты не стоят)."""
    zones: list[str]
    min_s: float = 5.0


class StaffCfg(Strict):
    uniform: UniformCfg | None = None
    zone: StaffZoneCfg | None = None
    # MVP-идентификация конкретного сотрудника — по секциям/сменам (см. объекты.owner)


# ---------- условия (общие для правил и состояний) ----------

class OccupancyCond(Strict):
    """Сколько людей (по роли) в зоне сейчас."""
    occupancy: str                   # id зоны
    who: Who = "customer"
    op: Op = ">="
    value: float = 1


class DwellCond(Strict):
    """Максимальное время пребывания среди людей, находящихся в зоне сейчас."""
    dwell_max: str
    who: Who = "customer"
    op: Op = ">="
    value: float


class ContactCond(Strict):
    """Сотрудник находился в зоне контакта ≥ min_s (с момента входа в текущее состояние)."""
    contact: str                     # id зоны контакта
    min_s: float = 3.0


class TimeInStateCond(Strict):
    time_in_state: Op
    value: float


class AllCond(Strict):
    all: list["Condition"]


class AnyCond(Strict):
    any: list["Condition"]


class NotCond(Strict):
    not_: "Condition" = Field(alias="not")
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


Condition = Union[OccupancyCond, DwellCond, ContactCond, TimeInStateCond,
                  AllCond, AnyCond, NotCond]
AllCond.model_rebuild(); AnyCond.model_rebuild(); NotCond.model_rebuild()


# ---------- правила (глобальные оповещения) ----------

class Rule(Strict):
    id: str
    when: Condition
    for_s: float = 0                 # условие держится непрерывно столько секунд
    cooldown_s: float = 300
    severity: Literal["info", "warning", "critical"] = "warning"
    message: str                     # шаблон: {zone_id.customer}, {zone_id.staff}, {duration}
    notify: list[Literal["console", "telegram"]] = ["console"]


# ---------- объекты с состояниями (стол, касса, покупатель-зона) ----------

class StateAlert(Strict):
    after_s: float
    severity: Literal["info", "warning", "critical"] = "warning"
    message: str
    notify: list[Literal["console", "telegram"]] = ["console"]


class StateDef(Strict):
    label: str | None = None
    alerts: list[StateAlert] = []
    counts_as_service_wait: bool = False   # время в этом состоянии = «ожидание обслуживания»


class Transition(Strict):
    from_: list[str] | str = Field(alias="from")
    to: str
    when: Condition
    for_s: float = 0
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class StateMachine(Strict):
    initial: str
    states: dict[str, StateDef]
    transitions: list[Transition]

    @model_validator(mode="after")
    def _check(self):
        names = set(self.states)
        if self.initial not in names:
            raise ValueError(f"initial '{self.initial}' не объявлен в states")
        for t in self.transitions:
            src = [t.from_] if isinstance(t.from_, str) else t.from_
            for s in [*src, t.to]:
                if s != "*" and s not in names:
                    raise ValueError(f"переход ссылается на неизвестное состояние '{s}'")
        return self


class TrackedObject(Strict):
    """Экземпляр шаблона: стол 5, касса 1, примерочная…

    `zones` подставляет реальные зоны в плейсхолдеры шаблона: в шаблоне пишем
    `$seat`, в объекте — `zones: {seat: table_5}`.
    """
    id: str
    machine: str
    camera: str
    zones: dict[str, str]
    owner: str | None = None         # кто отвечает (секция официанта / кассир смены)


# ---------- выход ----------

class TelegramCfg(Strict):
    enabled: bool = False
    token_env: str = "TG_BOT_TOKEN"
    chat_id_env: str = "TG_CHAT_ID"
    send_snapshot: bool = True


class FrigateCfg(Strict):
    url: str = "http://127.0.0.1:5000"     # внутренний API Frigate
    clip_before_s: float = 10
    clip_after_s: float = 10


class OutputCfg(Strict):
    dir: str = "out"
    annotated_video: bool = True
    telegram: TelegramCfg = TelegramCfg()


class Site(Strict):
    id: str
    name: str
    vertical: Literal["grocery", "pharmacy", "convenience", "fashion", "cafe",
                      "restaurant", "qsr", "fuel", "beauty", "gym", "bank",
                      "warehouse", "carwash", "other"]
    timezone: str = "Asia/Almaty"


class SiteConfig(Strict):
    site: Site
    cameras: list[Camera]
    detector: DetectorCfg = DetectorCfg()
    tracker: TrackerCfg = TrackerCfg()
    staff: StaffCfg = StaffCfg()
    rules: list[Rule] = []
    machines: dict[str, StateMachine] = {}
    objects: list[TrackedObject] = []
    output: OutputCfg = OutputCfg()
    frigate: FrigateCfg | None = None

    @model_validator(mode="after")
    def _refs(self):
        zone_ids = {z.id for c in self.cameras for z in c.zones}
        cams = {c.id for c in self.cameras}
        for o in self.objects:
            if o.machine not in self.machines:
                raise ValueError(f"объект {o.id}: нет машины состояний '{o.machine}'")
            if o.camera not in cams:
                raise ValueError(f"объект {o.id}: нет камеры '{o.camera}'")
            for v in o.zones.values():
                if v not in zone_ids:
                    raise ValueError(f"объект {o.id}: нет зоны '{v}'")
        if self.staff.zone:
            for z in self.staff.zone.zones:
                if z not in zone_ids:
                    raise ValueError(f"staff.zone: нет зоны '{z}'")
        for r in self.rules:
            for z in _cond_zones(r.when):
                if z not in zone_ids:
                    raise ValueError(f"правило {r.id}: нет зоны '{z}'")
        return self


def _cond_zones(c) -> list[str]:
    if isinstance(c, OccupancyCond): return [c.occupancy]
    if isinstance(c, DwellCond): return [c.dwell_max]
    if isinstance(c, ContactCond): return [c.contact]
    if isinstance(c, AllCond): return [z for x in c.all for z in _cond_zones(x)]
    if isinstance(c, AnyCond): return [z for x in c.any for z in _cond_zones(x)]
    if isinstance(c, NotCond): return _cond_zones(c.not_)
    return []


def load(path: str | Path) -> SiteConfig:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg = SiteConfig.model_validate(data)
    # относительные пути к видео — от корня проекта (родитель папки configs)
    root = path.resolve().parent.parent
    for cam in cfg.cameras:
        if "://" not in cam.source and not Path(cam.source).is_absolute():
            cam.source = str(root / cam.source)
    return cfg
