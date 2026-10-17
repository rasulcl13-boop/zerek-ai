# STG-NF на PoseLift — воспроизведение

Код STG-NF (Hirschorn & Avidan, ICCV 2023), адаптированный авторами PoseLift, распространяется по
CC BY-NC 4.0 и в репозиторий не включён. Он лежит в папке набора PoseLift на Google Drive
(ссылка — в https://github.com/TeCSAR-UNCC/PoseLift).

```bash
# 1. данные PoseLift (позы + метки) и код STG-NF — скрипт кладёт их в data/poselift
./scripts/get_data.sh poselift
# 2. код авторов → сюда, правки совместимости
rsync -a --exclude __pycache__ data/poselift/STG-NF/ research/stgnf/ && research/stgnf/patch_compat.sh
# 3. pickle → JSON в формате STG-NF (оси не переставляются, как у авторов) + маски
mkdir -p research/stgnf/data/PoseLift/gt/test_frame_mask research/stgnf/data/PoseLift/pose/{train,test}
cp data/poselift/Pickle_files/GT/*.npy research/stgnf/data/PoseLift/gt/test_frame_mask/
.venv/bin/python -I research/stgnf/prepare_poselift.py data/poselift/Pickle_files
# 4. обучение и оценка (параметры авторов по умолчанию; MPS не поддерживает float64 → CPU)
cd research/stgnf && for s in 0 1 2 3 4; do ../../.venv/bin/python run_stgnf.py --seed $s --device cpu --num_workers 0; done
```

Результаты по сидам — `results/seed*.json` (уже лежат в репозитории). Сравнение с нашим kNN на том
же протоколе — `tools/eval_poselift.py` и раздел PoseLift в корневом README.
