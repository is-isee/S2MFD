#!/bin/bash
# 人工拡散の弱いランは静止状態から始めると初期過渡で壊れる。
# ある程度緩和した状態を種にして始め直す。
cd /scr/a000/c0234hotta/Repository/S2MFD || exit 1
export MPLBACKEND=Agg PYTHONPATH=/scr/a000/c0234hotta/Repository/S2MFD
export OMP_NUM_THREADS=3 S2MFD_PARALLEL=3 S2MFD_PARFILE=parameters/rempel06_paper.py
PY=/scr/a000/c0234hotta/venv-paris/bin/python
SEED=results_rempel/seed_cs010.npz
for cs in 0.02 0.05; do
  tag="cs${cs/./}_seed"
  setsid nohup $PY -u runs/relax_scan.py 108 72 25 uniform_rotation "$tag" \
      sld_cs_factor=$cs init=$SEED > results_rempel/$tag.log 2>&1 < /dev/null &
  echo "  $tag (pid $!)"
done
