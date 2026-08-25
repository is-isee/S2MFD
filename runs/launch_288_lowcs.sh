#!/bin/bash
# 二重極限を詰める最後の 1 点: 288x192 で人工拡散を下げる。
# 216x144 では cs=0.02 まで安定だったので 288x192 でも通るはず
# (解像度が上がるほど von Neumann の中立点も上がる)。
# 288x192 の飽和状態を種にして過渡を飛ばす。
cd /scr/a000/c0234hotta/Repository/S2MFD || exit 1
export MPLBACKEND=Agg PYTHONPATH=/scr/a000/c0234hotta/Repository/S2MFD
export S2MFD_PARFILE=parameters/rempel06_paper.py
PY=/scr/a000/c0234hotta/venv-paris/bin/python
for cs in 0.05 0.02; do
  tag="s288x192_cs${cs/./}"
  OMP_NUM_THREADS=12 S2MFD_PARALLEL=12 \
  setsid nohup $PY -u runs/relax_scan.py 288 192 25 uniform_rotation "$tag" \
      sld_cs_factor=$cs init=results_rempel/seed288_cs010.npz \
      > results_rempel/$tag.log 2>&1 < /dev/null &
  echo "  $tag (12スレ, pid $!)"
done
