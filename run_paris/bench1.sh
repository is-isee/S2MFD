#!/bin/bash
cd /scr/a000/c0234hotta/Repository/S2MFD || exit 1
export MPLBACKEND=Agg OMP_NUM_THREADS=${T:-4} NUMBA_NUM_THREADS=${T:-4}
/scr/a000/c0234hotta/venv-paris/bin/python run_paris/bench_step.py "$1" "$2" "${3:-300}"
