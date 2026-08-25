#!/bin/bash
cd /scr/a000/c0234hotta/Repository/S2MFD || exit 1
export MPLBACKEND=Agg PYTHONPATH=/scr/a000/c0234hotta/Repository/S2MFD
for t in 1 2 4 8 16; do
  OMP_NUM_THREADS=$t NUMBA_NUM_THREADS=$t /scr/a000/c0234hotta/venv-paris/bin/python \
    runs/bench_step.py "$1" "$2" 300 | tail -1
done
