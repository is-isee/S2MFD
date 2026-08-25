#!/bin/bash
# 高解像度でのスレッド並列の効き。本番ランが動いている上で測るので
# 絶対値は競合込みだが、同じ背景負荷の下での**相対比較**は意味を持つ。
cd /scr/a000/c0234hotta/Repository/S2MFD || exit 1
export MPLBACKEND=Agg PYTHONPATH=/scr/a000/c0234hotta/Repository/S2MFD
PY=/scr/a000/c0234hotta/venv-paris/bin/python
for t in 1 2 4 8 16; do
  OMP_NUM_THREADS=$t S2MFD_PARALLEL=$t $PY runs/bench_cpu.py "$1" "$2" 150 2>/dev/null \
      | tail -1 | sed "s/^/P=$t  /"
done
