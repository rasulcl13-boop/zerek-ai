"""Движок событий: треки → зоны/линии/роли → правила и машины состояний.

Не знает ничего про видео и нейросети: на вход — время и список треков с
нормированными рамками. Поэтому его можно гонять синтетическими сценариями
(tests/) и одинаково использовать для продуктового, кафе и бутика.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from types import SimpleNamespace

from . import config as C

OPS = {">": lambda a, b: a > b, ">=": lambda a, b: a >= b, "<": lambda a, b: a < b,
       "<=": lambda a, b: a <= b, "==": lambda a, b: a == b, "!=": lambda a, b: a != b}


@dataclass
class TrackIn:
    id: int
    box: tuple[float, float, float, float]   # x1,y1,x2,y2 в долях кадра
    uniform_ratio: float | None = None        # доля «цвета формы» на торсе


@dataclass
class TrackState:
    id: int
    first_seen: float
    last_seen: float
    anchor: tuple[float, float]
    role: str = "customer"
    role_reason: str = ""
    uniform: list[float] = field(default_factory=list)
    staff_zone_s: float = 0.0
    zones: dict[str, float] = field(default_factory=dict)   # zone_id → время входа
    line_side: dict[str, int] = field(default_factory=dict)
    outside_since: dict[str, float] = field(default_factory=dict)   # зона → когда вышла точка


@dataclass
class Event:
    t: float
    type: str
    camera: str
    data: dict

    def as_row(self):
        return {"t": round(self.t, 3), "type": self.type, "camera": self.camera, **self.data}


def point_in_poly(p, poly) -> bool:
    x, y = p
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xi = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xi:
                inside = not inside
    return inside


def _side(a, b, p) -> float:
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def _line_side(line, p) -> int:
    """+1/−1 — по какую сторону линии точка (дальше гистерезиса), 0 — в полосе или вне отрезка."""
    (ax, ay), (bx, by) = line.start, line.end
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    u = ((p[0] - ax) * dx + (p[1] - ay) * dy) / L2          # проекция на отрезок
    if not -0.05 <= u <= 1.05:
        return 0
    d = _side(line.start, line.end, p) / math.sqrt(L2)      # знаковое расстояние
    return 1 if d > line.hysteresis else -1 if d < -line.hysteresis else 0


def segments_cross(p1, p2, a, b) -> bool:
    d1, d2 = _side(a, b, p1), _side(a, b, p2)
    d3, d4 = _side(p1, p2, a), _side(p1, p2, b)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


class ObjectRuntime:
    def __init__(self, obj: C.TrackedObject, machine: C.StateMachine, t0: float):
        self.obj, self.m = obj, machine
        self.state = machine.initial
        self.entered = t0
        self.fired: set[int] = set()
        self.contact_s: dict[str, float] = {}
        self.contact_staff: dict[str, set[int]] = {}
        self.pending: dict[int, float] = {}   # индекс перехода → с какого момента условие истинно

    def zone(self, ref: str) -> str:
        return self.obj.zones.get(ref[1:], ref) if ref.startswith("$") else ref


class Engine:
    def __init__(self, cfg: C.SiteConfig, presence_grace_s: float = 1.0, zone_exit_grace_s: float = 1.0):
        self.cfg = cfg
        self.grace = presence_grace_s
        self.zone_grace = zone_exit_grace_s   # выход из зоны засчитывается, если точка вне её дольше
        self.zones = {z.id: (cam.id, z) for cam in cfg.cameras for z in cam.zones}
        self.lines = {l.id: (cam.id, l) for cam in cfg.cameras for l in cam.lines}
        self.tracks: dict[str, dict[int, TrackState]] = {c.id: {} for c in cfg.cameras}
        self.rule_since: dict[str, float | None] = {r.id: None for r in cfg.rules}
        self.rule_last_fire: dict[str, float] = {}
        self.objects: list[ObjectRuntime] | None = None
        self.counters = {l: {"in": 0, "out": 0} for l in self.lines}
        self.last_t: dict[str, float] = {}
        self.anchor_mode = {c.id: c.anchor for c in cfg.cameras}

    # ---------- публичное ----------

    def step(self, t: float, camera: str, tracks: list[TrackIn]) -> list[Event]:
        ev: list[Event] = []
        dt = t - self.last_t.get(camera, t)
        self.last_t[camera] = t
        if self.objects is None:
            self.objects = [ObjectRuntime(o, self.cfg.machines[o.machine], t) for o in self.cfg.objects]
        self._update_tracks(t, dt, camera, tracks, ev)
        self._eval_rules(t, camera, ev)
        self._eval_objects(t, dt, camera, ev)
        return ev

    def flush(self, camera: str) -> list[Event]:
        """Конец записи: закрыть открытые пребывания в зонах и треки."""
        ev: list[Event] = []
        for tid, st in list(self.tracks[camera].items()):
            for zid in list(st.zones):
                self._exit(st.last_seen, camera, st, zid, ev)
            ev.append(Event(st.last_seen, "track_end", camera,
                            {"track": tid, "role": st.role, "life_s": round(st.last_seen - st.first_seen, 2),
                             "open": True}))
        self.tracks[camera].clear()
        return ev

    def zone_people(self, zone_id: str, who: str = "any") -> list[TrackState]:
        cam, _ = self.zones[zone_id]
        return [s for s in self.tracks[cam].values()
                if zone_id in s.zones and (who == "any" or s.role == who)]

    def snapshot(self, t: float) -> dict:
        """Состояние для оверлея/дашборда."""
        return {
            "zones": {z: {"customer": len(self.zone_people(z, "customer")),
                          "staff": len(self.zone_people(z, "staff"))} for z in self.zones},
            "lines": self.counters,
            "objects": {o.obj.id: {"state": o.state, "for_s": t - o.entered, "owner": o.obj.owner}
                        for o in (self.objects or [])},
        }

    # ---------- треки, роли, зоны, линии ----------

    def _update_tracks(self, t, dt, camera, tracks, ev):
        cam_tracks = self.tracks[camera]
        staff_cfg = self.cfg.staff
        for tr in tracks:
            x1, y1, x2, y2 = tr.box
            anchor = ((x1 + x2) / 2, y2 if self.anchor_mode[camera] == "bottom" else (y1 + y2) / 2)
            st = cam_tracks.get(tr.id)
            if st is None:
                st = cam_tracks[tr.id] = TrackState(tr.id, t, t, anchor)
                ev.append(Event(t, "track_start", camera, {"track": tr.id}))
            prev_anchor = st.anchor
            st.anchor, st.last_seen = anchor, t

            # линии: сторона трека с гистерезисом, проход засчитывается при смене стороны
            for lid, (lcam, line) in self.lines.items():
                if lcam != camera:
                    continue
                side = _line_side(line, anchor)
                if side == 0:
                    continue
                prev = st.line_side.get(lid)
                st.line_side[lid] = side
                if prev is None or prev == side:
                    continue
                direction = "in" if prev > 0 else "out"
                if line.swap_direction:
                    direction = "out" if direction == "in" else "in"
                if st.role == "customer":          # персонал не считаем в трафике
                    self.counters[lid][direction] += 1
                ev.append(Event(t, "line_cross", camera,
                                {"line": lid, "dir": direction, "track": tr.id, "role": st.role}))

            # зоны
            for zid, (zcam, zone) in self.zones.items():
                if zcam != camera:
                    continue
                inside = point_in_poly(anchor, zone.polygon)
                if inside:
                    st.outside_since.pop(zid, None)
                    if zid not in st.zones:
                        st.zones[zid] = t
                        ev.append(Event(t, "zone_enter", camera, {"zone": zid, "track": tr.id, "role": st.role}))
                elif zid in st.zones:
                    left = st.outside_since.setdefault(zid, t)
                    if t - left >= self.zone_grace:
                        self._exit(left, camera, st, zid, ev)
                        st.outside_since.pop(zid)

            # роль
            if tr.uniform_ratio is not None:
                st.uniform.append(tr.uniform_ratio)
                st.uniform = st.uniform[-30:]
            if staff_cfg.zone and any(z in st.zones for z in staff_cfg.zone.zones):
                st.staff_zone_s += dt
            if st.role != "staff":
                reason = None
                u = staff_cfg.uniform
                if u and len(st.uniform) >= u.min_samples and statistics.median(st.uniform) >= u.min_ratio:
                    reason = f"форма {statistics.median(st.uniform):.2f}"
                elif staff_cfg.zone and st.staff_zone_s >= staff_cfg.zone.min_s:
                    reason = f"staff-зона {st.staff_zone_s:.0f}с"
                if reason:
                    st.role, st.role_reason = "staff", reason
                    ev.append(Event(t, "role", camera, {"track": tr.id, "role": "staff", "reason": reason}))

        # пропавшие треки
        for tid in [k for k, s in cam_tracks.items() if t - s.last_seen > self.grace]:
            st = cam_tracks.pop(tid)
            for zid in list(st.zones):
                self._exit(st.last_seen, camera, st, zid, ev)
            ev.append(Event(t, "track_end", camera,
                            {"track": tid, "role": st.role, "life_s": round(st.last_seen - st.first_seen, 2)}))

    def _exit(self, t, camera, st, zid, ev):
        dwell = t - st.zones.pop(zid)
        ev.append(Event(t, "zone_exit", camera,
                        {"zone": zid, "track": st.id, "role": st.role, "dwell_s": round(dwell, 2)}))

    # ---------- условия ----------

    def _cond(self, c, t, obj: ObjectRuntime | None = None) -> bool:
        z = (lambda ref: obj.zone(ref)) if obj else (lambda ref: ref)
        if isinstance(c, C.OccupancyCond):
            return OPS[c.op](len(self.zone_people(z(c.occupancy), c.who)), c.value)
        if isinstance(c, C.DwellCond):
            ppl = self.zone_people(z(c.dwell_max), c.who)
            mx = max((t - p.zones[z(c.dwell_max)] for p in ppl), default=0.0)
            return OPS[c.op](mx, c.value)
        if isinstance(c, C.ContactCond):
            return obj is not None and obj.contact_s.get(z(c.contact), 0.0) >= c.min_s
        if isinstance(c, C.TimeInStateCond):
            return obj is not None and OPS[c.time_in_state](t - obj.entered, c.value)
        if isinstance(c, C.AllCond):
            return all(self._cond(x, t, obj) for x in c.all)
        if isinstance(c, C.AnyCond):
            return any(self._cond(x, t, obj) for x in c.any)
        if isinstance(c, C.NotCond):
            return not self._cond(c.not_, t, obj)
        raise TypeError(c)

    def _ctx(self, t, obj: ObjectRuntime | None = None, duration: float = 0.0):
        def dwell(zid):
            return _fmt_s(max((t - p.zones[zid] for p in self.zone_people(zid, "customer")), default=0))
        ctx = {zid: SimpleNamespace(customer=len(self.zone_people(zid, "customer")),
                                    staff=len(self.zone_people(zid, "staff")),
                                    any=len(self.zone_people(zid)), dwell=dwell(zid))
               for zid in self.zones}
        ctx["duration"] = _fmt_s(duration)
        if obj:
            ctx.update(object=obj.obj.id, owner=obj.obj.owner or "—", state=obj.state,
                       wait=_fmt_s(t - obj.entered))
            for ref, zid in obj.obj.zones.items():
                ctx.setdefault(ref, ctx[zid])
        return ctx

    # ---------- правила ----------

    def _eval_rules(self, t, camera, ev):
        for r in self.cfg.rules:
            if not self._rule_on_camera(r, camera):
                continue
            if self._cond(r.when, t):
                since = self.rule_since[r.id]
                if since is None:
                    since = self.rule_since[r.id] = t
                last = self.rule_last_fire.get(r.id, -math.inf)
                if t - since >= r.for_s and t - last >= r.cooldown_s:
                    self.rule_last_fire[r.id] = t
                    msg = r.message.format_map(self._ctx(t, duration=t - since))
                    ev.append(Event(t, "alert", camera, {"rule": r.id, "severity": r.severity,
                                                         "message": msg, "notify": r.notify}))
            else:
                self.rule_since[r.id] = None

    def _rule_on_camera(self, r: C.Rule, camera: str) -> bool:
        zs = C._cond_zones(r.when)
        return not zs or any(self.zones[z][0] == camera for z in zs)

    # ---------- машины состояний ----------

    def _eval_objects(self, t, dt, camera, ev):
        for o in self.objects:
            if o.obj.camera != camera:
                continue
            # накопление контакта: любой сотрудник в зоне объекта
            for zid in set(o.obj.zones.values()):
                staff = self.zone_people(zid, "staff")
                if staff:
                    o.contact_s[zid] = o.contact_s.get(zid, 0.0) + dt
                    o.contact_staff.setdefault(zid, set()).update(s.id for s in staff)

            # переходы
            for i, tr in enumerate(o.m.transitions):
                src = [tr.from_] if isinstance(tr.from_, str) else tr.from_
                if (o.state not in src and "*" not in src) or tr.to == o.state:
                    o.pending.pop(i, None)
                    continue
                if self._cond(tr.when, t, o):
                    since = o.pending.setdefault(i, t)
                    if t - since >= tr.for_s:
                        self._change_state(o, tr.to, t, camera, ev)
                        break
                else:
                    o.pending.pop(i, None)

            # оповещения по времени в состоянии
            sdef = o.m.states[o.state]
            for k, a in enumerate(sdef.alerts):
                if k not in o.fired and t - o.entered >= a.after_s:
                    o.fired.add(k)
                    msg = a.message.format_map(self._ctx(t, o))
                    ev.append(Event(t, "alert", camera, {"rule": f"{o.obj.id}:{o.state}:{k}",
                                                         "object": o.obj.id, "owner": o.obj.owner,
                                                         "severity": a.severity, "message": msg,
                                                         "notify": a.notify}))

    def _change_state(self, o: ObjectRuntime, new: str, t, camera, ev):
        old, spent = o.state, t - o.entered
        staff_ids = sorted({i for s in o.contact_staff.values() for i in s})
        ev.append(Event(t, "state", camera, {
            "object": o.obj.id, "owner": o.obj.owner, "from": old, "to": new,
            "spent_s": round(spent, 2), "service_wait": o.m.states[old].counts_as_service_wait,
            "staff_tracks": staff_ids,
        }))
        o.state, o.entered = new, t
        o.fired.clear(); o.pending.clear(); o.contact_s.clear(); o.contact_staff.clear()


def _fmt_s(s: float) -> str:
    s = int(round(s))
    return f"{s // 60} мин {s % 60:02d} с" if s >= 60 else f"{s} с"
