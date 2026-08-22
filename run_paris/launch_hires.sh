#!/bin/bash
# 高解像度ランを、低解像度の飽和状態を補間した種から始める (過渡を飛ばす)。
cd /scr/a000/c0234hotta/Repository/S2MFD || exit 1
export MPLBACKEND=Agg PYTHONPATH=/scr/a000/c0234hotta/Repository/S2MFD
export OMP_NUM_THREADS=3 S2MFD_PARALLEL=3 S2MFD_PARFILE=parameters/rempel06_paper.py
PY=/scr/a000/c0234hotta/venv-paris/bin/python
run () {  # nx ny cs seed years
  local nx=$1 ny=$2 cs=$3 seed=$4 yr=$5
  local tag="s${nx}x${ny}_cs${cs/./}"
  setsid nohup $PY -u run_paris/relax_scan.py "$nx" "$ny" "$yr" uniform_rotation "$tag" \
      sld_cs_factor=$cs init=results_rempel/$seed \
      > results_rempel/$tag.log 2>&1 < /dev/null &
  echo "  $tag (種 $seed, ${yr}yr, pid $!)"
}
case "$1" in
  X) run 216 144 0.30 seed_cs030_216x144.npz 30
     run 216 144 0.10 seed_cs010_216x144.npz 30
     run 216 144 0.05 seed_cs010_216x144.npz 30 ;;
  Y) run 288 192 0.30 seed_cs030_288x192.npz 30
     run 288 192 0.10 seed_cs010_288x192.npz 30 ;;
  Z) run 144 96 0.30 seed_cs030_144x96.npz 30
     run 144 96 0.10 seed_cs010_144x96.npz 30
     run 144 96 0.05 seed_cs010_144x96.npz 30 ;;
  *) echo "X/Y/Z"; exit 1;;
esac
