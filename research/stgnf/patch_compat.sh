#!/bin/sh
# Правки совместимости кода STG-NF авторов PoseLift с современными библиотеками (логика не меняется):
#   matplotlib ≥ 3.6: стиль 'seaborn-ticks' переименован в 'seaborn-v0_8-ticks'
#   NumPy ≥ 1.24:     псевдонимы np.int / np.float / np.bool / np.object удалены
set -e
cd "$(dirname "$0")"
sed -i.bak "s/plt.style.use('seaborn-ticks')/plt.style.use('seaborn-v0_8-ticks')/" utils/pose_utils.py && rm -f utils/pose_utils.py.bak
grep -rlE "np\.(int|float|bool|object)\b" --include='*.py' . | while read -r f; do
  perl -pi -e 's/\bnp\.(int|float|bool|object)\b(?![0-9_])/$1/g' "$f"
done
rm -f gen_data.py   # вызывает AlphaPose через os.system — не нужен, позы уже в датасете
echo "patched"
