#!/bin/bash
# 残る系統的なずれが「解像度」で説明できるかを直接確かめる。
#
# 60 年完走した 108x72 の結果 (2026-08-23):
#   運動学的参照解  周期 17.9 年 (論文 19)、max B_phi 1.50 T (1.28)、
#                   max B_r 0.0121 T (0.01)、DR 0.321 (0.27)
#   非運動学的      周期 18.0-18.1 年 (18) で一致、磁場と交換項は 1 割以内、
#                   DR だけ 1.13-1.15 倍
#
# 磁場なしの収束性検証では DR が 108x72 で 0.322、収束値 0.267 と分かって
# いる (比 1.19)。運動学的参照解のずれもこれで説明できるなら、**解像度を
# 上げると周期が 17.9 -> 19 に、max B_phi が 1.50 -> 1.28 に動くはず**。
#
# 周期を測るには 2 サイクル (36 年) は要る。144x96 で 40 年。
cd /scr/a000/c0234hotta/Repository/S2MFD || exit 1
export MPLBACKEND=Agg PYTHONPATH=/scr/a000/c0234hotta/Repository/S2MFD
export S2MFD_PARFILE=parameters/rempel06_paper.py
PY=/scr/a000/c0234hotta/venv-paris/bin/python
OMP_NUM_THREADS=5 S2MFD_PARALLEL=5 \
setsid nohup $PY -u runs/dynamo7.py 144 96 0 42 res144_kin kinematic \
    magnetic_buoyancy=1 bseed=3000 sld_cs_factor=0.30 alpha0=12.5 \
    init=results_rempel/s144x96_cs030/state.npz \
    > results_rempel/res144_kin.log 2>&1 < /dev/null &
echo "  res144_kin (144x96 運動学的, 42年, 5スレ, pid $!)"
OMP_NUM_THREADS=5 S2MFD_PARALLEL=5 \
setsid nohup $PY -u runs/dynamo7.py 144 96 0 42 res144_a125 full \
    magnetic_buoyancy=1 bseed=3000 sld_cs_factor=0.30 alpha0=12.5 \
    init=results_rempel/s144x96_cs030/state.npz \
    > results_rempel/res144_a125.log 2>&1 < /dev/null &
echo "  res144_a125 (144x96 非運動学的, 42年, 5スレ, pid $!)"
