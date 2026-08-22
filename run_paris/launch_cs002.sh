#!/bin/bash
# cs=0.02 は S=0.30 (線形中立点 0.314 のすぐ内側) でも発散した。
# 線形解析が捉えない非線形な不安定なので、安全率をさらに下げて限界を探す。
cd /scr/a000/c0234hotta/Repository/S2MFD || exit 1
export MPLBACKEND=Agg PYTHONPATH=/scr/a000/c0234hotta/Repository/S2MFD
export OMP_NUM_THREADS=3 S2MFD_PARALLEL=3 S2MFD_PARFILE=parameters/rempel06_paper.py
PY=/scr/a000/c0234hotta/venv-paris/bin/python
for S in 0.15 0.08; do
  tag="cs002_S${S/./}"
  setsid nohup $PY -u run_paris/relax_scan.py 108 72 25 uniform_rotation "$tag" \
      sld_cs_factor=0.02 cfl_safety=$S > results_rempel/$tag.log 2>&1 < /dev/null &
  echo "  $tag (S=$S, pid $!)"
done
