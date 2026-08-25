#!/bin/bash
cd /scr/a000/c0234hotta/Repository/S2MFD || exit 1
export MPLBACKEND=Agg PYTHONPATH=/scr/a000/c0234hotta/Repository/S2MFD OMP_NUM_THREADS=8
PY=${PY:-python}
for s in 0 1 2 3 4; do
  S2MFD_PARALLEL=$s $PY runs/bitcheck.py "$1" "$2" "${3:-200}" 2>/dev/null | tail -1
done
