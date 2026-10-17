"""Обучение и оценка STG-NF на PoseLift — как train_eval.py авторов, плюс сохранение покадровых оценок.

    cd research/stgnf && ../../.venv/bin/python run_stgnf.py --seed 0 --device mps --num_workers 0

Параметры модели и данных — по умолчанию из args.py авторов (seg_len 24, 8 эпох, K=8, R=3 …).
Сохраняет results/seed<N>.npz: покадровые gt (1 = кража) и сглаженную аномальность по клипам.
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve

from args import create_exp_dirs, init_parser, init_sub_args
from dataset import get_dataset_and_loader
from models.STG_NF.model_pose import STG_NF
from models.training import Trainer
from utils.data_utils import trans_list
from utils.optim_init import init_optimizer, init_scheduler
from utils.scoring_utils import get_dataset_scores, smooth_scores
from utils.train_utils import init_model_params


class _NullWriter:
    def add_scalar(self, *a, **k):
        pass


def metrics(gt_anom: np.ndarray, anomaly: np.ndarray) -> dict:
    """Обе конвенции: «как у авторов» (положительный класс — норма) и «кража — положительный»."""
    out = {}
    for name, y, s in (("authors_normal_positive", 1 - gt_anom, -anomaly), ("theft_positive", gt_anom, anomaly)):
        p, r, _ = precision_recall_curve(y, s)
        fpr, tpr, _ = roc_curve(y, s)
        i = np.nanargmin(np.abs((1 - tpr) - fpr))
        out[name] = {"AUC-ROC": round(100 * roc_auc_score(y, s), 2), "AUC-PR": round(100 * auc(r, p), 2),
                     "EER": round(float(fpr[i]), 3), "PR_baseline": round(100 * float(y.mean()), 1)}
    return out


def main():
    parser = init_parser()
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(0); torch.manual_seed(args.seed)
    args, _ = init_sub_args(args)
    args.ckpt_dir = create_exp_dirs(args.exp_dir, dirmap=args.dataset)
    dataset, loader = get_dataset_and_loader(args, trans_list=trans_list, only_test=False)
    model = STG_NF(**init_model_params(args, dataset))
    trainer = Trainer(args, model, loader["train"], loader["test"],
                      optimizer_f=init_optimizer(args.model_optimizer, lr=args.model_lr),
                      scheduler_f=init_scheduler(args.model_sched, lr=args.model_lr, epochs=args.epochs))
    trainer.train(log_writer=_NullWriter())
    normality = trainer.test()

    # подсчёт — функциями авторов: оценка окна в центр, минимум по людям, inf → max, сглаживание
    gt_arr, sc_arr = get_dataset_scores(normality, dataset["test"].metadata, args=args, save_results=False)
    sc_arr = smooth_scores(sc_arr)
    gt_normal = np.concatenate(gt_arr)           # у авторов 1 = норма
    norm_sc = np.concatenate(sc_arr)
    gt_anom = (1 - gt_normal).astype(int)
    res = {"seed": args.seed, "device": args.device, "epochs": args.epochs, "seg_len": args.seg_len,
           "test_frames": int(len(gt_anom)), "train_segments": int(len(dataset["train"])),
           **metrics(gt_anom, -norm_sc)}
    Path("results").mkdir(exist_ok=True)
    np.savez(f"results/seed{args.seed}.npz", gt_anom=gt_anom, anomaly=-norm_sc,
             lens=np.array([len(g) for g in gt_arr]))
    Path(f"results/seed{args.seed}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    sys.exit(main())
