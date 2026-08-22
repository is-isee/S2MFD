#!/bin/bash
# 高解像度ランは飽和状態を補間した種から始め、スレッド数も格子に合わせる。
#
# 実測 (astana, 本番ランが動いている状態での相対比較):
#   216x144: P=1 29.6ms  P=3 14.4  P=4 12.0  P=6 8.9  P=8 7.4  P=12 7.0
#   288x192: P=1 71.5ms  P=2 35.0  P=4 21.2  P=8 14.0  P=16 11.2
# 3 -> 8 で 1.9 倍。効率は 8 スレッドで 50-64% とまだ実用域。
# 144x96 (13824 セル) は 108x144 (15552) より小さいので 3 のままでよい。
cd /scr/a000/c0234hotta/Repository/S2MFD || exit 1
export MPLBACKEND=Agg PYTHONPATH=/scr/a000/c0234hotta/Repository/S2MFD
export S2MFD_PARFILE=parameters/rempel06_paper.py
PY=/scr/a000/c0234hotta/venv-paris/bin/python
run () {  # nx ny cs seed years threads
  local nx=$1 ny=$2 cs=$3 seed=$4 yr=$5 th=$6
  local tag="s${nx}x${ny}_cs${cs/./}"
  OMP_NUM_THREADS=$th S2MFD_PARALLEL=$th \
  setsid nohup $PY -u run_paris/relax_scan.py "$nx" "$ny" "$yr" uniform_rotation "$tag" \
      sld_cs_factor=$cs init=results_rempel/$seed \
      > results_rempel/$tag.log 2>&1 < /dev/null &
  echo "  $tag (${th}スレッド, ${yr}yr, pid $!)"
}
case "$1" in
  X) run 216 144 0.30 seed_cs030_216x144.npz 30 8
     run 216 144 0.10 seed_cs010_216x144.npz 30 8
     run 216 144 0.05 seed_cs010_216x144.npz 30 8 ;;
  Y) run 288 192 0.30 seed_cs030_288x192.npz 30 12
     run 288 192 0.10 seed_cs010_288x192.npz 30 12 ;;
  Z) run 144 96 0.30 seed_cs030_144x96.npz 30 3
     run 144 96 0.10 seed_cs010_144x96.npz 30 3
     run 144 96 0.05 seed_cs010_144x96.npz 30 3 ;;
  *) echo "X/Y/Z"; exit 1;;
esac
