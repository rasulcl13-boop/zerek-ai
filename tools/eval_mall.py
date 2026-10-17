"""Mall (CUHK): сколько людей в зоне — детектор против ручной разметки голов.

    python tools/eval_mall.py [--model rfdetr-medium]

2000 кадров 640×480 с камеры ТЦ, 13–53 человека в кадре. Стандартный сплит литературы:
кадры 1–800 — подбор, 801–2000 — проверка. Порог уверенности подбираем на 1–800.
Считаем человека, если «голова» (верх рамки + 10% высоты) попадает в ROI разметки.
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
from storevision import config as C  # noqa: E402
from storevision.perception import PersonDetector  # noqa: E402

D = ROOT / "data/mall/mall_dataset"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="rfdetr-medium")
    ap.add_argument("--slice", action="store_true", help="SAHI-нарезка кадра 2×2 (supervision.InferenceSlicer)")
    ap.add_argument("--step", type=int, default=1, help="брать каждый N-й кадр")
    a = ap.parse_args()
    gt = sio.loadmat(D / "mall_gt.mat")["count"].ravel()
    roi = sio.loadmat(D / "perspective_roi.mat")["roi"][0][0]["mask"].astype(bool)
    tag = f"{a.model}{'_slice' if a.slice else ''}_c05_s{a.step}"
    cache = ROOT / f"out/cache/mall_{tag}.npz"
    frames_idx = list(range(1, 2001, a.step))
    if cache.exists():
        dets = dict(np.load(cache, allow_pickle=True))["dets"].tolist()
    else:
        det = PersonDetector(C.DetectorCfg(model=a.model, confidence=0.05))
        if a.slice:
            import supervision as sv
            slicer = sv.InferenceSlicer(callback=lambda im: det(im), slice_wh=(360, 280), overlap_wh=(80, 40),
                                        iou_threshold=0.5)
        dets = []
        for n, i in enumerate(frames_idx, 1):
            img = cv2.imread(str(D / "frames" / f"seq_{i:06d}.jpg"))
            d = slicer(img) if a.slice else det(img)
            dets.append(np.hstack([d.xyxy, d.confidence[:, None]]) if len(d) else np.zeros((0, 5)))
            if n % 200 == 0:
                print(f"  {n}/{len(frames_idx)}", flush=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, dets=np.array(dets, dtype=object))

    h, w = roi.shape

    def counts(th):
        out = []
        for d in dets:
            d = d[d[:, 4] >= th]
            hx = ((d[:, 0] + d[:, 2]) / 2).astype(int).clip(0, w - 1)
            hy = (d[:, 1] + 0.1 * (d[:, 3] - d[:, 1])).astype(int).clip(0, h - 1)
            out.append(int(roi[hy, hx].sum()))
        return np.array(out)

    gt = gt[np.array(frames_idx) - 1]
    idx = np.array(frames_idx)
    tr, te = idx <= 800, idx > 800
    sweep = {round(th, 2): float(np.abs(counts(th)[tr] - gt[tr]).mean()) for th in np.arange(0.05, 0.75, 0.05)}
    best = min(sweep, key=sweep.get)
    c = counts(best)
    err = c[te] - gt[te]
    res = {
        "model": a.model + (" + SAHI 2×2" if a.slice else ""), "frame_step": a.step,
        "threshold_from_train": best,
        "test_frames": int(len(err)),
        "MAE": round(float(np.abs(err).mean()), 2),
        "MSE(RMSE)": round(float(np.sqrt((err ** 2).mean())), 2),
        "mean_bias": round(float(err.mean()), 2),
        "relative_error_%": round(float(100 * np.abs(err).sum() / gt[te].sum()), 1),
        "gt_mean": round(float(gt[te].mean()), 1),
        "by_crowd": {f"{lo}-{hi}": round(float(np.abs(err[(gt[te] >= lo) & (gt[te] < hi)]).mean()), 2)
                     for lo, hi in [(13, 25), (25, 35), (35, 54)]},
        "train_sweep_MAE": {k: round(v, 2) for k, v in sweep.items()},
    }
    print(json.dumps(res, ensure_ascii=False, indent=1))
    (ROOT / "out/eval").mkdir(parents=True, exist_ok=True)
    (ROOT / f"out/eval/mall_{tag}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
