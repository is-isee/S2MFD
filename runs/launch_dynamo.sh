#!/bin/bash
# Rempel 2006 表 1 の再現。飽和した緩和状態から磁場を入れて 60 年回す。
#   p_kin : 図 3 の運動学的参照解 (alpha クエンチングあり、周期 19yr, 1.28T)
#   p_a125: 表 1 列 3 (alpha0 = 0.125 m/s)
#   p_a250: 表 1 列 5 (alpha0 = 0.25  m/s)
#   p_a500: 表 1 列 7 (alpha0 = 0.5   m/s)
# CFL 安全率は dynamo7.py が von Neumann の中立点から自動決定する。
cd /scr/a000/c0234hotta/Repository/S2MFD || exit 1
export MPLBACKEND=Agg PYTHONPATH=/scr/a000/c0234hotta/Repository/S2MFD
export OMP_NUM_THREADS=3 S2MFD_PARALLEL=3 S2MFD_PARFILE=parameters/rempel06_paper.py
PY=/scr/a000/c0234hotta/venv-paris/bin/python
BASE=${BASE:-m108x72_cs030}; CS=${CS:-0.30}
INIT=results_rempel/$BASE/state.npz
[ -f "$INIT" ] || { echo "緩和済み状態がない: $INIT"; exit 1; }
echo "初期値 $INIT  sld_cs_factor=$CS"
launch () {
  local tag=$1; shift; local mode=$1; shift
  setsid nohup $PY -u runs/dynamo7.py 108 72 0 60 "$tag" "$mode" \
      magnetic_buoyancy=1 bseed=3000 sld_cs_factor=$CS init=$INIT "$@" \
      > results_rempel/$tag.log 2>&1 < /dev/null &
  echo "  $tag (pid $!)"
}
launch ${P}kin  kinematic alpha0=12.5
launch ${P}a125 full      alpha0=12.5
launch ${P}a250 full      alpha0=25.0
launch ${P}a500 full      alpha0=50.0
