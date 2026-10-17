#!/bin/sh
# Загрузка данных для демо и оценок. Наборы не входят в репозиторий (лицензии, размер).
#   ./scripts/get_data.sh            — видео для демо + CAVIAR + Mall + PoseLift
#   ./scripts/get_data.sh merl       — + MERL Shopping (1,7 ГБ)
#   ./scripts/get_data.sh <часть>    — videos | caviar | mall | merl | poselift | frigate
# Лицензии: Pexels — Pexels License; CAVIAR, Mall, MERL — только для исследований; PoseLift — см. репозиторий авторов.
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
want() { [ -z "$1" ] && return 0; for p in $PARTS; do [ "$p" = "$1" ] && return 0; done; return 1; }
PARTS="${*:-videos caviar mall poselift frigate}"

if want videos; then
  mkdir -p data/videos
  for id_file in 4249560/4249560-uhd_3840_2160_25fps 36635591/15532413_3840_2160_30fps; do
    id=${id_file%%/*}
    [ -f data/videos/pexels-$id.mp4 ] || curl -fL -A "Mozilla/5.0" -o data/videos/pexels-$id.mp4 \
      "https://videos.pexels.com/video-files/$id_file.mp4"
  done
  for f in grocery-store people-walking; do
    [ -f data/videos/$f.mp4 ] || curl -fL -o data/videos/$f.mp4 \
      "https://media.roboflow.com/supervision/video-examples/$f.mp4"
  done
fi

if want caviar; then
  mkdir -p data/caviar
  B=https://groups.inf.ed.ac.uk/vision/DATASETS/CAVIAR/CAVIARDATA2
  for s in TwoEnterShop1front:f2es1gt TwoEnterShop2front:f2es2gt TwoEnterShop3front:f2es3gt \
           TwoLeaveShop1front:f2ls1gt OneLeaveShopReenter1front:folsr1gt \
           EnterExitCrossingPaths1front:feecp1gt WalkByShop1front:fwbs1gt; do
    seq=${s%%:*}; gt=${s##*:}
    [ -f data/caviar/$seq.mpg ] || curl -fL -o data/caviar/$seq.mpg "$B/$seq/$seq.mpg"
    [ -f data/caviar/$gt.xml ] || curl -fL -o data/caviar/$gt.xml "$B/$seq/$gt.xml"
  done
fi

if want mall; then
  mkdir -p data/mall
  if [ ! -d data/mall/mall_dataset ]; then
    curl -fL -o data/mall/mall_dataset.zip https://personal.ie.cuhk.edu.hk/~ccloy/files/datasets/mall_dataset.zip
    (cd data/mall && unzip -q mall_dataset.zip && rm mall_dataset.zip)
  fi
fi

if want merl; then
  mkdir -p data/merl
  M=https://ftp.merl.com/pub/tmarks/MERL_Shopping_Dataset
  for z in Labels Videos; do
    [ -d data/merl/${z}_MERL_Shopping_Dataset ] && continue
    curl -fL -o data/merl/$z.zip "$M/${z}_MERL_Shopping_Dataset.zip"
    (cd data/merl && unzip -q $z.zip && rm -rf $z.zip __MACOSX)
  done
fi

if want poselift; then
  mkdir -p data/poselift
  $PY -m pip show gdown >/dev/null 2>&1 || $PY -m pip install -q gdown
  [ -d data/poselift/Pickle_files ] || .venv/bin/gdown --folder \
    "https://drive.google.com/drive/folders/1aEkENZlVE4ZvF_BZXV1VJOwuiXQq6trn" -O data/poselift
fi

if want frigate; then
  # демо-«камеры» для Frigate: те же ролики в H.264 1280 px (go2rtc крутит их по кругу)
  mkdir -p frigate/media/demo frigate/storage
  [ -f frigate/media/demo/pexels-4249560.mp4 ] || ffmpeg -v error -y -i data/videos/pexels-4249560.mp4 \
    -vf scale=1280:-2 -c:v libx264 -preset veryfast -g 25 -bf 0 -an frigate/media/demo/pexels-4249560.mp4
  [ -f frigate/media/demo/grocery-store.mp4 ] || ffmpeg -v error -y -i data/videos/grocery-store.mp4 \
    -vf scale=1280:-2 -c:v libx264 -preset veryfast -g 25 -bf 0 -an frigate/media/demo/grocery-store.mp4
  [ -f frigate/media/demo/caviar-front.mp4 ] || ffmpeg -v error -y -i data/caviar/TwoEnterShop1front.mpg \
    -c:v libx264 -preset veryfast -g 25 -bf 0 -an frigate/media/demo/caviar-front.mp4
fi
echo "готово: $PARTS"
