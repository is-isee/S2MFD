#!/bin/bash
# cs020 の発散が時間刻みのせいか物理かを切り分ける: CFL 安全率を半分にして再投入。
cd /scr/a000/c0234hotta/Repository/S2MFD || exit 1
export MPLBACKEND=Agg PYTHONPATH=/scr/a000/c0234hotta/Repository/S2MFD
export OMP_NUM_THREADS=3 S2MFD_PARALLEL=3 S2MFD_PARFILE=parameters/rempel06_paper.py
PY=/scr/a000/c0234hotta/venv-paris/bin/python
setsid nohup $PY -u run_paris/relax_scan.py 108 72 15 uniform_rotation cs020_cfl025 \
    sld_cs_factor=0.20 cfl_safety=0.25 \
    > results_rempel/cs020_cfl025.log 2>&1 < /dev/null &
echo "  cs020_cfl025 (pid $!)"
