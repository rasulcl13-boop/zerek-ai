"""Pickle PoseLift → JSON в формате STG-NF (логика convert_anno.py авторов, оси не переставляем)."""
import json
import pickle
import sys
from pathlib import Path

SRC = Path(sys.argv[1])          # data/poselift/Pickle_files
DST = Path(__file__).resolve().parent / "data/PoseLift/pose"

for split_src, split_dst in (("Train", "train"), ("Test", "test")):
    n = 0
    for f in sorted((SRC / split_src).glob("*.pkl")):
        ann = pickle.loads(f.read_bytes())
        ids = sorted({k for d in ann.values() for k in (d or {})})
        out = {}
        for pid in ids:
            pd_ = {str(fr): {"keypoints": [float(x) for x in ann[fr][pid][1].flatten()], "scores": None}
                   for fr in ann if ann[fr] and pid in ann[fr]}
            out[str(pid)] = pd_
        if out:
            (DST / split_dst / f"{f.stem}_alphapose_tracked_person.json").write_text(json.dumps(out))
            n += 1
    print(split_dst, n)
