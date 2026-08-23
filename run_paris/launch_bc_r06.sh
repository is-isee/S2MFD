#!/bin/bash
# 対照実験: 磁場の下部境界条件。
#
# Rempel (2006) §2.2:
#   "B_Phi vanishes at both radial boundaries, while A vanishes at the inner
#    boundary, and the poloidal field is assumed to be radial at the top
#    boundary."
#
# つまり下部境界 (r=0.65R) は A=0 かつ **B_Phi=0**。ところが
# rempel06_paper.py は defaults.py の boundary_condition_type='vertical' を
# 継承しており、下部が**完全導体** (A=0, d(rB_Phi)/dr=0) になっていた。
# 論文どおりの 'R06' は boundary_condition.py に実装済みなのに、どの
# パラメタファイルからも使われていなかった (2026-08-23 に発見)。
#
# 走っている v2_kin が 'vertical' の対照そのものなので、同じ設定で
# boundary_condition_type=R06 だけ変えた 1 本を出せば差が見える。
cd /scr/a000/c0234hotta/Repository/S2MFD || exit 1
export MPLBACKEND=Agg PYTHONPATH=/scr/a000/c0234hotta/Repository/S2MFD
export S2MFD_PARFILE=parameters/rempel06_paper.py
PY=/scr/a000/c0234hotta/venv-paris/bin/python
OMP_NUM_THREADS=3 S2MFD_PARALLEL=3 \
setsid nohup $PY -u run_paris/dynamo7.py 108 72 0 32 bcR06_kin kinematic \
    magnetic_buoyancy=1 bseed=3000 sld_cs_factor=0.30 alpha0=12.5 \
    boundary_condition_type=R06 \
    init=results_rempel/m108x72_cs030/state.npz \
    > results_rempel/bcR06_kin.log 2>&1 < /dev/null &
echo "  bcR06_kin (下部 B_phi=0, 3スレ, pid $!)  <- 対照は v2_kin"
