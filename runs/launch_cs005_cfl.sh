#!/bin/bash
cd /scr/a000/c0234hotta/Repository/S2MFD || exit 1
export MPLBACKEND=Agg PYTHONPATH=/scr/a000/c0234hotta/Repository/S2MFD
export OMP_NUM_THREADS=3 S2MFD_PARALLEL=3 S2MFD_PARFILE=parameters/rempel06_paper.py
PY=/scr/a000/c0234hotta/venv-paris/bin/python
setsid nohup $PY -u runs/relax_scan.py 108 72 25 uniform_rotation cs005_cfl0125 \
    sld_cs_factor=0.05 cfl_safety=0.125 > results_rempel/cs005_cfl0125.log 2>&1 < /dev/null &
echo "  cs005_cfl0125 (pid $!)"
