#!/bin/bash
cd /scr/a000/c0234hotta/Repository/S2MFD || exit 1
export MPLBACKEND=Agg PYTHONPATH=/scr/a000/c0234hotta/Repository/S2MFD OMP_NUM_THREADS=8
PY=${PY:-python}
for rep in 1 2 3; do
  for s in 0 1; do
    S2MFD_PARALLEL=$s $PY run_paris/bench_cpu.py "$1" "$2" 500 2>/dev/null | tail -1 | sed "s/^/rep$rep P=$s  /"
  done
done
