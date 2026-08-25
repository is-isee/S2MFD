#!/bin/bash
# 空いている資源で検証を厚くする。
#   F: 定常到達の確認 (80 年ラン)。40 年で足りているかを判定するための対照。
#   G: 人工拡散の下限探索 (cs=0.01/0.02)。安全率は自動なので純粋に
#      「拡散が足りるか」だけを見る。
#   H: 緯度解像度だけを振る (動径 108 固定)。CFL は動径律速なので dt が
#      変わらず、緯度分解能の効果を分離できる。
#   I: 全球 [0,pi] との照合 (半球化が定常解でも等価であることの実測)。
cd /scr/a000/c0234hotta/Repository/S2MFD || exit 1
export MPLBACKEND=Agg PYTHONPATH=/scr/a000/c0234hotta/Repository/S2MFD
export OMP_NUM_THREADS=3 S2MFD_PARALLEL=3 S2MFD_PARFILE=parameters/rempel06_paper.py
PY=/scr/a000/c0234hotta/venv-paris/bin/python
run () {  # tag nx ny years extra...
  local tag=$1 nx=$2 ny=$3 yr=$4; shift 4
  setsid nohup $PY -u runs/relax_scan.py "$nx" "$ny" "$yr" uniform_rotation "$tag" \
      "$@" > results_rempel/$tag.log 2>&1 < /dev/null &
  echo "  $tag (pid $!)"
}
case "$1" in
  F) run long80_cs010 108 72 80 sld_cs_factor=0.10
     run long80_cs030 108 72 80 sld_cs_factor=0.30 ;;
  G) run m108x72_cs001 108 72 25 sld_cs_factor=0.01
     run m108x72_cs002 108 72 25 sld_cs_factor=0.02 ;;
  H) run lat108x48  108 48  25 sld_cs_factor=0.10
     run lat108x144 108 144 25 sld_cs_factor=0.10
     run lat108x216 108 216 25 sld_cs_factor=0.10 ;;
  I) run full108x144 108 144 25 sld_cs_factor=0.10 thmax=3.141592653589793 ;;
  *) echo "F/G/H/I"; exit 1;;
esac
