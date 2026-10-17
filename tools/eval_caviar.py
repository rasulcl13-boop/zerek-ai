"""Точность счётчика входа/выхода на настоящей CCTV (CAVIAR, ручная разметка).

    python tools/eval_caviar.py [--video]
GT: вход = объект, у которого в разметке есть situation «shop enter»; выход — «shop exit».
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from storevision import config as C  # noqa: E402
from storevision.run import run_cfg  # noqa: E402

SEQ = {"TwoEnterShop1front": "f2es1gt.xml", "TwoEnterShop2front": "f2es2gt.xml",
       "TwoEnterShop3front": "f2es3gt.xml", "TwoLeaveShop1front": "f2ls1gt.xml",
       "OneLeaveShopReenter1front": "folsr1gt.xml", "EnterExitCrossingPaths1front": "feecp1gt.xml",
       "WalkByShop1front": "fwbs1gt.xml"}


# В WalkByShop другой зум камеры: порог магазина ниже. Как и в жизни — у каждой камеры своя разметка.
LINE_OVERRIDE = {"WalkByShop1front": {"start": (0.18, 0.70), "end": (0.95, 0.70)}}


def gt_counts(xml: Path) -> tuple[int, int]:
    s = xml.read_text()
    enter, exit_ = set(), set()
    for m in re.finditer(r'<object id="(\d+)">.*?<situation[^>]*>([^<]+)<', s, re.S):
        oid, sit = m.groups()
        (enter if sit == "shop enter" else exit_ if sit == "shop exit" else set()).add(oid)
    return len(enter), len(exit_)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--slice", choices=["off", "on"], default=None,
                    help="переопределить SAHI-нарезку камеры (по умолчанию — как в конфиге)")
    a = ap.parse_args()
    base = C.load(ROOT / "configs/caviar_entrance.yaml")
    if a.slice == "on" and not base.cameras[0].slicing:
        base.cameras[0].slicing = C.SlicingCfg()
    elif a.slice == "off":
        base.cameras[0].slicing = None
    mode = "slice" if base.cameras[0].slicing else "noslice"
    print(f"SAHI: {base.cameras[0].slicing}")
    rows, tot = [], [0, 0, 0, 0]
    for seq, gt in SEQ.items():
        cfg = base.model_copy(deep=True)
        cfg.cameras[0].source = str(ROOT / "data/caviar" / f"{seq}.mpg")
        for k, v in LINE_OVERRIDE.get(seq, {}).items():
            setattr(cfg.cameras[0].lines[0], k, v)
        rep = run_cfg(cfg, video=a.video, quiet=True, out=ROOT / "out/caviar_eval" / mode / seq)
        door = rep["traffic"].get("door", {"in": 0, "out": 0})
        gi, go = gt_counts(ROOT / "data/caviar" / gt)
        rows.append((seq, gi, door["in"], go, door["out"]))
        for k, v in enumerate((gi, door["in"], go, door["out"])):
            tot[k] += v
    print(f"\n{'сценарий':32} {'вход GT':>8} {'вход':>6} {'выход GT':>9} {'выход':>6}")
    for r in rows:
        mark = "" if (r[1], r[3]) == (r[2], r[4]) else "  ✗"
        print(f"{r[0]:32} {r[1]:8} {r[2]:6} {r[3]:9} {r[4]:6}{mark}")
    print(f"{'ИТОГО':32} {tot[0]:8} {tot[1]:6} {tot[2]:9} {tot[3]:6}")
    err = sum(abs(r[2] - r[1]) + abs(r[4] - r[3]) for r in rows)   # по каждому ролику — без взаимозачёта
    n = tot[0] + tot[2]
    print(f"ошибка по роликам (лишние + пропущенные): {err} из {n} событий ({100 * err / max(1, n):.0f}%)")


if __name__ == "__main__":
    main()
