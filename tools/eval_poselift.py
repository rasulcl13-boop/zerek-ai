"""PoseLift (WACV 2025): обнаружение краж по позам как аномалий — базовая модель + сравнение со статьёй.

    python tools/eval_poselift.py

Данные: позы людей с 6 камер реального магазина (без видео), протокол STG-NF:
train — только обычные покупки, test — покупки и кражи с покадровой маской.
Модель (без обучения нейросети): окна поз по T кадров → нормализация (центр — бёдра/плечи,
масштаб — рост скелета) → признаки (координаты, скорости, «кисть у бедра/пояса») → PCA →
расстояние до k ближайших нормальных окон = оценка аномальности. Оценка кадра — максимум по
людям и окнам, покрывающим кадр. Метрики как в статье: AUC-ROC, AUC-PR, EER по всем кадрам теста.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data/poselift/Pickle_files"     # полный набор: Train 104, Test 47, GT 47
T, STRIDE = 12, 2           # окно 0.8 с при 15 к/с: тестовые ролики короткие (32–200 кадров)
if "--T" in sys.argv:       # проверка устойчивости: --T 24 (seg_len авторов STG-NF)
    T = int(sys.argv[sys.argv.index("--T") + 1])
PAPER = {"STG-NF": (67.46, 84.06, 0.39), "TSGAD": (63.35, 39.31, 0.41), "GEPC": (60.61, 50.38, 0.38)}
L_SH, R_SH, L_HIP, R_HIP, L_WR, R_WR = 5, 6, 11, 12, 9, 10


def load_people(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """person → (номера кадров, [n,17,3] в порядке x, y, conf).

    В pickle точки лежат как (y, x, conf) — сверено с рамкой [x1, y1, x2, y2] того же человека.
    """
    d = pickle.loads(path.read_bytes())
    per: dict[int, list] = {}
    for f, people in d.items():
        for pid, (_box, kp) in (people or {}).items():
            kp = np.asarray(kp, dtype=np.float32)
            per.setdefault(pid, []).append((int(f), kp[:, [1, 0, 2]]))
    out = {}
    for pid, rows in per.items():
        rows.sort(key=lambda r: r[0])
        out[pid] = (np.array([r[0] for r in rows]), np.stack([r[1] for r in rows]))
    return out


def normalize(kp: np.ndarray) -> np.ndarray:
    xy, c = kp[..., :2], kp[..., 2:]
    center = xy[:, [L_SH, R_SH, L_HIP, R_HIP]].mean(1, keepdims=True)
    h = (xy[..., 1].max(1) - xy[..., 1].min(1)).reshape(-1, 1, 1) + 1e-3
    return np.concatenate([(xy - center) / h, c], -1)


def windows(frames: np.ndarray, kp: np.ndarray, stride: int = STRIDE):
    """Окна по непрерывным кускам трека."""
    if len(frames) < T:
        return [], []
    n = normalize(kp)
    feats, spans = [], []
    for s in range(0, len(frames) - T + 1, stride):
        f = frames[s:s + T]
        if f[-1] - f[0] > T * 2:                # разрыв трека
            continue
        w = n[s:s + T]
        xy = w[..., :2]
        vel = np.diff(xy, axis=0)
        hip = xy[:, [L_HIP, R_HIP]].mean(1)
        wr = xy[:, [L_WR, R_WR]]
        d_wr_hip = np.linalg.norm(wr - hip[:, None], axis=-1)          # кисть у кармана/пояса
        d_wr_torso = np.linalg.norm(wr - xy[:, [L_SH, R_SH, L_HIP, R_HIP]].mean(1)[:, None], axis=-1)
        v = np.concatenate([xy.mean(0).ravel(), xy.std(0).ravel(), np.abs(vel).mean(0).ravel(),
                                     d_wr_hip.mean(0), d_wr_hip.min(0), d_wr_torso.mean(0),
                                     w[..., 2].mean(0)])
        if not np.isfinite(v).all():            # пропуски в позах — окно пропускаем
            continue
        feats.append(v)
        spans.append((f[0], f[-1]))
    return feats, spans


def authors_protocol(te_files, scorer, featurize):
    """Покадровые оценки ровно как utils/scoring_utils.py STG-NF (PoseLift):
    окна с шагом 1, оценка — в центральный кадр окна; по людям — минимум нормальности;
    кадры без оценки = +inf нормальности → глобальный максимум; сглаживание gaussian σ=1..6 подряд."""
    gts, norms = [], []
    for f in te_files:
        mask_p = D / "GT" / f"{f.stem}.npy"
        if not mask_p.exists():
            continue
        gt = np.load(mask_p).astype(int)
        per_person = []
        for fr, kp in load_people(f).values():
            feats, spans = windows(fr, kp, stride=1)
            row = np.full(len(gt), np.inf)
            if feats:
                z = scorer(featurize(np.array(feats)))
                for (a, _b), v in zip(spans, z):
                    c = a + T // 2
                    if 0 <= c < len(gt):
                        row[c] = -v                    # нормальность = −аномальность
            per_person.append(row)
        clip = np.min(np.stack(per_person), axis=0) if per_person else np.full(len(gt), np.inf)
        gts.append(gt); norms.append(clip)
    allv = np.concatenate(norms)
    finite_max = allv[np.isfinite(allv)].max()
    out = []
    for n in norms:
        n = np.where(np.isposinf(n), finite_max, n)
        for sig in range(1, 7):
            n = gaussian_filter1d(n, sigma=sig)
        out.append(n)
    return np.concatenate(gts), -np.concatenate(out)        # gt (1 = кража), аномальность


def both_conventions(gt_anom, anomaly):
    """Как run_stgnf.py: «как у авторов» (положительный класс — норма, AUC-PR = auc(recall, precision))
    и «кража — положительный класс»."""
    out = {}
    for name, y, s in (("authors_normal_positive", 1 - gt_anom, -anomaly), ("theft_positive", gt_anom, anomaly)):
        p, r, _ = precision_recall_curve(y, s)
        fpr, tpr, _ = roc_curve(y, s)
        i = np.nanargmin(np.abs((1 - tpr) - fpr))
        out[name] = {"AUC-ROC": round(100 * roc_auc_score(y, s), 2), "AUC-PR": round(100 * auc(r, p), 2),
                     "EER": round(float(fpr[i]), 3), "PR_baseline": round(100 * float(y.mean()), 1)}
    return out


def main():
    tr_files = sorted((D / "Train").glob("*.pkl"))
    te_files = sorted((D / "Test").glob("*.pkl"))
    Xtr, cams = [], []
    for f in tr_files:
        for fr, kp in load_people(f).values():
            w = windows(fr, kp)[0]
            Xtr += w; cams += [f.stem.split("_")[0]] * len(w)
    Xtr, cams = np.array(Xtr), np.array(cams)
    sc = StandardScaler().fit(Xtr)
    pca = PCA(n_components=0.95, random_state=0).fit(sc.transform(Xtr))
    Ztr = pca.transform(sc.transform(Xtr))
    K = 5

    def make_scorer(Z):
        nn = NearestNeighbors(n_neighbors=K + 1).fit(Z)
        ref = nn.kneighbors(Z)[0][:, 1:].mean(1)                     # leave-one-out на трейне
        mu, sd = ref.mean(), ref.std() + 1e-9
        return lambda X: (nn.kneighbors(X, n_neighbors=K)[0].mean(1) - mu) / sd

    global_scorer = make_scorer(Ztr)
    cam_scorers = {c: make_scorer(Ztr[cams == c]) for c in set(cams) if (cams == c).sum() >= 200}

    results = {}
    for variant in ("global", "per_camera"):
        scores, labels = [], []
        for f in te_files:
            name = f.stem
            mask_p = D / "GT" / f"{name}.npy"
            if not mask_p.exists():
                continue
            gt = np.load(mask_p).astype(int)
            scorer = cam_scorers.get(name.split("_")[0], global_scorer) if variant == "per_camera" else global_scorer
            s = np.full(len(gt), -np.inf)
            for fr, kp in load_people(f).values():
                feats, spans = windows(fr, kp)
                if not feats:
                    continue
                z = scorer(pca.transform(sc.transform(np.array(feats))))
                for (a, b), v in zip(spans, z):
                    a, b = max(0, a), min(len(gt) - 1, b)
                    s[a:b + 1] = np.maximum(s[a:b + 1], v)
            s[np.isinf(s)] = np.nan
            fill = np.nanmin(s) if np.isfinite(s).any() else 0.0
            scores.append(np.nan_to_num(s, nan=fill)); labels.append(gt)
        results[variant] = (np.concatenate(labels), np.concatenate(scores), len(scores))

    featurize = lambda X: pca.transform(sc.transform(X))
    gt_ap, an_ap = authors_protocol(te_files, global_scorer, featurize)
    authors = both_conventions(gt_ap, an_ap)
    (ROOT / "out/eval").mkdir(parents=True, exist_ok=True)
    np.savez(ROOT / f"out/eval/poselift_knn_T{T}_authors_protocol.npz", gt_anom=gt_ap, anomaly=an_ap)

    def metrics(y, sc_):
        fpr, tpr, _ = roc_curve(y, sc_)
        return {"AUC-ROC": round(100 * roc_auc_score(y, sc_), 2),
                "AUC-PR": round(100 * average_precision_score(y, sc_), 2),
                "EER": round(float(fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]), 2)}

    y = results["global"][0]
    res = {"model": "kNN по окнам поз (без нейросети)", "window_frames": T,
           "train_windows": int(len(Xtr)), "train_cameras": sorted(set(cams.tolist())),
           "pca_components": int(pca.n_components_), "test_videos": results["global"][2],
           "test_frames": int(len(y)), "theft_frame_share": round(float(y.mean()), 3),
           "ours_global": metrics(*results["global"][:2]),
           "ours_per_camera": metrics(*results["per_camera"][:2]),
           "ours_global_authors_protocol": authors,
           "paper_benchmarks": {k: {"AUC-ROC": v[0], "AUC-PR": v[1], "EER": v[2]} for k, v in PAPER.items()},
           "note": "Ориентировочно: статья считает по своему протоколу; здесь — все доступные видео теста."}
    print(json.dumps(res, ensure_ascii=False, indent=1))
    (ROOT / "out/eval").mkdir(parents=True, exist_ok=True)
    (ROOT / f"out/eval/poselift_T{T}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    sys.exit(main())
