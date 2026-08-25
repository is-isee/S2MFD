#!/bin/bash
# 下部境界の違いが**放射層の磁束の溜まり方と極性反転**にどう効くかを見る。
#
# 放射層 (0.65-0.71R) は eta_c = 1e5 m^2/s しかないので拡散時間が
# L^2/eta = 516 年 = 周期の 29 倍。**閉じた境界 (対称) だと数十サイクル分の
# 古い極性を溜め込む。** Rempel (2006) が下部で B_Phi = 0 を課しているのは
# これを抜くためかもしれない (堀田先生の見立て、2026-08-23)。
#
# R06 (反対称) は磁場の人工拡散フラックスを境界で開けないと発散する
# (magnetic_open_boundary=1)。物理パラメタは論文のまま。
#
# dynamo7.py の hist に列 20 (符号つき磁束) と 21 (|B_phi| の積分) を追加した
# ので、溜まっているか反転しているかが直接見える。
cd /scr/a000/c0234hotta/Repository/S2MFD || exit 1
export MPLBACKEND=Agg PYTHONPATH=/scr/a000/c0234hotta/Repository/S2MFD
export S2MFD_PARFILE=parameters/rempel06_paper.py
PY=/scr/a000/c0234hotta/venv-paris/bin/python
launch () {  # tag 追加引数...
  local tag=$1; shift
  OMP_NUM_THREADS=3 S2MFD_PARALLEL=3 \
  setsid nohup $PY -u runs/dynamo7.py 108 72 0 45 "$tag" kinematic \
      magnetic_buoyancy=1 bseed=3000 sld_cs_factor=0.30 alpha0=12.5 \
      init=results_rempel/m108x72_cs030/state.npz "$@" \
      > results_rempel/$tag.log 2>&1 < /dev/null &
  echo "  $tag (3スレ, pid $!) $*"
}
launch deep_vert_kin boundary_condition_type=vertical
launch deep_r06_kin  boundary_condition_type=R06 magnetic_open_boundary=1
