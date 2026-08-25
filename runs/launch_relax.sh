#!/bin/bash
# Rempel 2006 参照モデル (流体のみ) の緩和。論文の表 1 列 2: DR = 0.27
cd /scr/a000/c0234hotta/Repository/S2MFD || exit 1
export MPLBACKEND=Agg PYTHONPATH=/scr/a000/c0234hotta/Repository/S2MFD
export OMP_NUM_THREADS=3 S2MFD_PARALLEL=3 S2MFD_PARFILE=parameters/rempel06_paper.py
PY=/scr/a000/c0234hotta/venv-paris/bin/python
mkdir -p results_rempel
setsid nohup $PY -u runs/relax_scan.py 108 72 40 uniform_rotation paper_relax \
    > results_rempel/paper_relax.log 2>&1 < /dev/null &
echo "launched pid=$!"
