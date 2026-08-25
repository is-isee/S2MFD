#!/bin/bash
# poloidal_mag の引数修正 (B_theta が 2 倍だった) 後のダイナモ再投入。
# 旧ラン (p_*) はローレンツ力が過大な状態で飽和しているので、その最終状態は
# 使えない。流体緩和の飽和状態 (m108x72_cs030, 40 年で dDR/dt=+0.0001) から
# 新規に始める。
cd /scr/a000/c0234hotta/Repository/S2MFD || exit 1
export MPLBACKEND=Agg PYTHONPATH=/scr/a000/c0234hotta/Repository/S2MFD
export OMP_NUM_THREADS=3 S2MFD_PARALLEL=3 S2MFD_PARFILE=parameters/rempel06_paper.py
PY=/scr/a000/c0234hotta/venv-paris/bin/python
INIT=results_rempel/m108x72_cs030/state.npz
launch () {
  local tag=$1; shift; local mode=$1; shift
  setsid nohup $PY -u runs/dynamo7.py 108 72 0 60 "$tag" "$mode" \
      magnetic_buoyancy=1 bseed=3000 sld_cs_factor=0.30 init=$INIT "$@" \
      > results_rempel/$tag.log 2>&1 < /dev/null &
  echo "  $tag (pid $!)"
}
launch v2_kin  kinematic alpha0=12.5     # 図 3 の運動学的参照解
launch v2_a125 full      alpha0=12.5     # 表 1 列 3
launch v2_a250 full      alpha0=25.0     # 表 1 列 5
launch v2_a500 full      alpha0=50.0     # 表 1 列 7
