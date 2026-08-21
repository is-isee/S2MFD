#!/bin/bash
cd /scr/a000/c0234hotta/Repository/S2MFD || exit 1
export MPLBACKEND=Agg PYTHONPATH=/scr/a000/c0234hotta/Repository/S2MFD
for t in 1 2 3 4 8; do
  OMP_NUM_THREADS=$t NUMBA_NUM_THREADS=$t /scr/a000/c0234hotta/venv-paris/bin/python \
    run_paris/bench_cpu.py "$1" "$2" 400 | tail -1
done
