#!/bin/bash
# 人工拡散 (SLD) の強さに対する差動回転の依存性。
# 論文は人工拡散を持たない (MacCormack の交互風上/風下差分に内在する数値散逸で
# 代用) ので、sld_cs_factor -> 0 の極限で DR が論文の 0.27 に近づくかを見る。
cd /scr/a000/c0234hotta/Repository/S2MFD || exit 1
export MPLBACKEND=Agg PYTHONPATH=/scr/a000/c0234hotta/Repository/S2MFD
export OMP_NUM_THREADS=3 S2MFD_PARALLEL=3 S2MFD_PARFILE=parameters/rempel06_paper.py
PY=/scr/a000/c0234hotta/venv-paris/bin/python
for cs in 0.15 0.20 0.40 0.50; do
  tag="cs${cs/./}"
  setsid nohup $PY -u run_paris/relax_scan.py 108 72 15 uniform_rotation "$tag" \
      sld_cs_factor=$cs > results_rempel/$tag.log 2>&1 < /dev/null &
  echo "  $tag (cs_factor=$cs, pid $!)"
done
