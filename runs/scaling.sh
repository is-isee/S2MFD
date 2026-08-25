#!/bin/bash
cd /scr/a000/c0234hotta/Repository/S2MFD || exit 1
export MPLBACKEND=Agg PYTHONPATH=/scr/a000/c0234hotta/Repository/S2MFD
export OMP_NUM_THREADS=8
PY=${PY:-python}
for s in 0 2 3 4 6; do
  S2MFD_PARALLEL=$s $PY runs/bench_cpu.py "$1" "$2" 400 2>/dev/null | tail -1 | sed "s/^/S2MFD_PARALLEL=$s  /"
done
