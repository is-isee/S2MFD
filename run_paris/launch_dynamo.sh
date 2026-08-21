#!/bin/bash
# Rempel 2006 表 1 の再現。緩和済み状態 (results_rempel/paper_relax/state.npz) から
# 磁場を入れて 60 年回す。北半球のみ・0.985 RSUN・108x72・margin=2。
#
# 表 1 の列との対応 (すべて magnetic buoyancy ON):
#   列 3: alpha0 = 0.125 m/s = 12.5 cm/s
#   列 5: alpha0 = 0.25  m/s = 25   cm/s
#   列 7: alpha0 = 0.5   m/s = 50   cm/s
# 図 3 (運動学的参照解) は alpha クエンチングあり。'kinematic' を渡すと
# dynamo7.py が自動で alpha_quenching=True にする。
cd /scr/a000/c0234hotta/Repository/S2MFD || exit 1
export MPLBACKEND=Agg PYTHONPATH=/scr/a000/c0234hotta/Repository/S2MFD
export OMP_NUM_THREADS=3 S2MFD_PARALLEL=3 S2MFD_PARFILE=parameters/rempel06_paper.py
PY=/scr/a000/c0234hotta/venv-paris/bin/python
INIT=results_rempel/paper_relax/state.npz
[ -f "$INIT" ] || { echo "緩和済み状態がない: $INIT"; exit 1; }

launch () {  # tag  mode  extra...
  local tag=$1; shift; local mode=$1; shift
  setsid nohup $PY -u run_paris/dynamo7.py 108 72 0 60 "$tag" "$mode" \
      magnetic_buoyancy=1 bseed=3000 init=$INIT "$@" \
      > results_rempel/$tag.log 2>&1 < /dev/null &
  echo "  $tag (pid $!)"
}
echo "投入:"
launch p_kin  kinematic alpha0=12.5      # 図 3: 周期 19yr, max B_phi 1.28T
launch p_a125 full      alpha0=12.5      # 表 1 列 3
launch p_a250 full      alpha0=25.0      # 表 1 列 5
launch p_a500 full      alpha0=50.0      # 表 1 列 7
