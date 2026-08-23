#!/bin/bash
# res144_kin を 60 年まで延長する。
#
# 2026-08-24: 同じ時間窓 (18-38 年) で比べると、周期は解像度でほとんど
# 動かない (108x72 で 16.90 年、144x96 で 16.98 年)。一方 v2_kin の反転
# 間隔は 16.8 -> 18.0 と**時間とともに伸びる**ので、周期が論文の 19 年に
# 近づくのは飽和の効果らしい。
#
# 飽和した状態で比べるには v2_kin と同じ 60 年まで要る。42 年の
# final_state.npz から 18 年ぶん再開する (時刻も引き継がれる)。
cd /scr/a000/c0234hotta/Repository/S2MFD || exit 1
export MPLBACKEND=Agg PYTHONPATH=/scr/a000/c0234hotta/Repository/S2MFD
export S2MFD_PARFILE=parameters/rempel06_paper.py
PY=/scr/a000/c0234hotta/venv-paris/bin/python
OMP_NUM_THREADS=5 S2MFD_PARALLEL=5 \
setsid nohup $PY -u run_paris/dynamo7.py 144 96 0 18 res144_kin_b kinematic \
    magnetic_buoyancy=1 sld_cs_factor=0.30 alpha0=12.5 \
    init=results_rempel/res144_kin/final_state.npz \
    > results_rempel/res144_kin_b.log 2>&1 < /dev/null &
echo "  res144_kin_b (144x96 運動学的, +18年 = 通算 60年, 5スレ, pid $!)"
