"""MERL Shopping: «рука у полки» по позе (RF-DETR Keypoint) против ручной разметки действий.

    python tools/eval_merl.py [--train 6] [--test 10] [--fps 3]

Камера строго сверху, полка — верх кадра. Разметка: 5 действий по кадрам. Положительный кадр =
взаимодействие с полкой = «тянется к полке» ∪ «убирает руку» ∪ «рука в полке» (действия 1–3).
Признак кадра — самая верхняя уверенная кисть (COCO 9/10), нормированная по высоте кадра.
Порог подбирается на обучающих субъектах (1–20), качество — на тестовых (27–41), как в сплите авторов.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
V = ROOT / "data/merl/Videos_MERL_Shopping_Dataset"
L = ROOT / "data/merl/Labels_MERL_Shopping_Dataset"
CACHE = ROOT / "out/cache/merl"
WRISTS = (9, 10)


def gt_mask(name: str, n: int) -> np.ndarray:
    tl = sio.loadmat(L / f"{name}_label.mat")["tlabs"].ravel()
    m = np.zeros(n, bool)
    for a in (0, 1, 2):                       # reach, retract, hand-in-shelf
        for s, e in np.asarray(tl[a]).reshape(-1, 2):
            m[int(s) - 1:int(e)] = True
    return m


_model = None


def features(name: str, step: int) -> tuple[np.ndarray, np.ndarray, int]:
    """(индексы кадров, мин. y кисти в долях кадра или nan, число кадров)."""
    f = CACHE / f"{name}_s{step}.npz"
    if f.exists():
        z = np.load(f)
        return z["idx"], z["wy"], int(z["n"])
    global _model
    if _model is None:
        import rfdetr
        _model = rfdetr.RFDETRKeypointPreview()
    cap = cv2.VideoCapture(str(V / f"{name}_crop.mp4"))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx, wy, i = [], [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % step == 0:
            k = _model.predict(fr[:, :, ::-1].copy(), threshold=0.3)
            y = np.nan
            if len(k.xy):
                conf = k.confidence[:, WRISTS]
                ys = k.xy[:, WRISTS, 1][conf >= 0.3]
                if len(ys):
                    y = float(ys.min() / fr.shape[0])
            idx.append(i); wy.append(y)
        i += 1
    cap.release()
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(f, idx=np.array(idx), wy=np.array(wy), n=i)
    return np.array(idx), np.array(wy), i


def smooth(x: np.ndarray) -> np.ndarray:
    """Медиана по ±1 сэмплу (nan → 1.0, т.е. «руки у полки нет»)."""
    x = np.nan_to_num(x, nan=1.0)
    p = np.pad(x, 1, mode="edge")
    return np.median(np.stack([p[:-2], p[1:-1], p[2:]]), axis=0)


def prf(pred, gt):
    tp = (pred & gt).sum(); fp = (pred & ~gt).sum(); fn = (~pred & gt).sum()
    p = tp / max(1, tp + fp); r = tp / max(1, tp + fn)
    return p, r, 2 * p * r / max(1e-9, p + r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=6)
    ap.add_argument("--test", type=int, default=10)
    ap.add_argument("--fps", type=float, default=3)
    a = ap.parse_args()
    step = round(30 / a.fps)
    vids = sorted(p.name.replace("_crop.mp4", "") for p in V.glob("*_crop.mp4"))
    subj = lambda v: int(v.split("_")[0])
    train = [v for v in vids if subj(v) <= 20][::3][:a.train]       # по одному сеансу на субъекта
    test = [v for v in vids if subj(v) >= 27][::2][:a.test]

    def collect(names):
        X, Y, per = [], [], {}
        for v in names:
            idx, wy, n = features(v, step)
            g = gt_mask(v, n)[idx]
            s = smooth(wy)
            X.append(s); Y.append(g); per[v] = (s, g)
            print(f"  {v}: {len(idx)} кадров, доля «у полки» по разметке {g.mean():.0%}", flush=True)
        return np.concatenate(X), np.concatenate(Y), per

    print("обучение (подбор порога):", train)
    Xtr, Ytr, _ = collect(train)
    grid = np.arange(0.30, 0.70, 0.01)
    f1s = [prf(Xtr < t, Ytr)[2] for t in grid]
    th = float(grid[int(np.argmax(f1s))])
    print("тест:", test)
    Xte, Yte, per = collect(test)
    p, r, f = prf(Xte < th, Yte)
    base = Yte.mean()
    res = {"threshold_wrist_y": round(th, 2), "train_F1": round(max(f1s), 3),
           "test_frames": int(len(Yte)), "test_precision": round(float(p), 3),
           "test_recall": round(float(r), 3), "test_F1": round(float(f), 3),
           "baseline_always_yes_F1": round(float(2 * base / (1 + base)), 3),
           "positive_share": round(float(base), 3),
           "per_video_F1": {v: round(float(prf(s < th, g)[2]), 3) for v, (s, g) in per.items()},
           "train_videos": train, "test_videos": test, "sample_fps": a.fps}
    print(json.dumps(res, ensure_ascii=False, indent=1))
    (ROOT / "out/eval").mkdir(parents=True, exist_ok=True)
    (ROOT / "out/eval/merl.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
