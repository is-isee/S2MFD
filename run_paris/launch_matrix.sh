#!/bin/bash
# 収束性検証のための (解像度 x 人工拡散) 格子。
#
# 人工拡散の実効係数は kappa_SLD = 0.5 * c * dx なので dx に比例する。
# したがって cs_factor を固定したまま解像度を上げるだけで kappa_SLD -> 0 に
# なる。cs_factor スキャンは「同じ格子での感度」を測る別の軸。
#
# **CFL 安全率は指定しない。** relax_scan.py が von Neumann の中立点 S0 の
# 0.9 倍を自動で選ぶ (S2MFD/physics/stability.py)。S0 は解像度とともに上がり
# (72x48 で 0.274、288x192 で 0.506)、人工拡散が弱い領域では cs_factor に
# よらず一定になる (危険モードが 4-8 セルで SLD が効かないため)。
#
# 使い方: bash run_paris/launch_matrix.sh <A|B|C|D|E>
cd /scr/a000/c0234hotta/Repository/S2MFD || exit 1
export MPLBACKEND=Agg PYTHONPATH=/scr/a000/c0234hotta/Repository/S2MFD
export OMP_NUM_THREADS=3 S2MFD_PARALLEL=3 S2MFD_PARFILE=parameters/rempel06_paper.py
PY=/scr/a000/c0234hotta/venv-paris/bin/python
run () {  # nx ny cs years
  local nx=$1 ny=$2 cs=$3 yr=$4
  local tag="m${nx}x${ny}_cs${cs/./}"
  setsid nohup $PY -u run_paris/relax_scan.py "$nx" "$ny" "$yr" uniform_rotation "$tag" \
      sld_cs_factor=$cs > results_rempel/$tag.log 2>&1 < /dev/null &
  echo "  $tag (cs=$cs ${yr}yr, pid $!)"
}
case "$1" in
  A) run 108 72 0.30 40 ; run 108 72 0.10 40 ;;              # 本ラン
  B) run 108 72 0.03 25 ; run 108 72 0.05 25 ; run 108 72 0.15 25 ; run 108 72 0.20 25 ;;
  C) run 72 48 0.10 25 ; run 144 96 0.10 25 ; run 216 144 0.10 25 ;;
  D) run 72 48 0.30 25 ; run 144 96 0.30 25 ; run 216 144 0.30 25 ;;
  E) run 72 48 0.05 25 ; run 144 96 0.05 25 ; run 216 144 0.05 25 ; run 288 192 0.10 25 ;;
  *) echo "グループ A/B/C/D/E を指定"; exit 1;;
esac
