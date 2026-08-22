#!/bin/bash
# 磁場の SLD フィルタが磁場振幅を抑えているかの対照実験。
# Rempel 2006 には磁場への人工拡散がないので、切ったときに磁場が
# 論文値 (0.735R で 1.2 T) に近づくかを見る。
# 飽和した p_a125 の最終状態から磁場ごと再開するので過渡なし。
cd /scr/a000/c0234hotta/Repository/S2MFD || exit 1
export MPLBACKEND=Agg PYTHONPATH=/scr/a000/c0234hotta/Repository/S2MFD
export OMP_NUM_THREADS=3 S2MFD_PARALLEL=3 S2MFD_PARFILE=parameters/rempel06_paper.py
PY=/scr/a000/c0234hotta/venv-paris/bin/python
setsid nohup $PY -u run_paris/dynamo7.py 108 72 0 40 nofilt_a125 full \
    magnetic_buoyancy=1 sld_cs_factor=0.30 alpha0=12.5 magnetic_filter=0 \
    init=results_rempel/p_a125/final_state.npz \
    > results_rempel/nofilt_a125.log 2>&1 < /dev/null &
echo "  nofilt_a125 (磁場フィルタなし, pid $!)"
setsid nohup $PY -u run_paris/dynamo7.py 108 72 0 40 nofilt_a250 full \
    magnetic_buoyancy=1 sld_cs_factor=0.30 alpha0=25.0 magnetic_filter=0 \
    init=results_rempel/p_a250/final_state.npz \
    > results_rempel/nofilt_a250.log 2>&1 < /dev/null &
echo "  nofilt_a250 (磁場フィルタなし, pid $!)"
