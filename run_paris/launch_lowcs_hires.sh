#!/bin/bash
# 二重極限を詰める: 216x144 の飽和状態を種に、人工拡散をさらに下げる。
# 108x72 では cs=0.02 が発散したが、216x144 は物理粘性の拡散数が大きく
# von Neumann 中立点も高い (S0: 0.274 -> 0.398) ので、より低い cs まで通る
# 可能性がある。安全率は relax_scan.py が自動で決める。
cd /scr/a000/c0234hotta/Repository/S2MFD || exit 1
export MPLBACKEND=Agg PYTHONPATH=/scr/a000/c0234hotta/Repository/S2MFD
export S2MFD_PARFILE=parameters/rempel06_paper.py
PY=/scr/a000/c0234hotta/venv-paris/bin/python
for cs in 0.02 0.03; do
  tag="s216x144_cs${cs/./}"
  OMP_NUM_THREADS=8 S2MFD_PARALLEL=8 \
  setsid nohup $PY -u run_paris/relax_scan.py 216 144 20 uniform_rotation "$tag" \
      sld_cs_factor=$cs init=results_rempel/seed216_cs010.npz \
      > results_rempel/$tag.log 2>&1 < /dev/null &
  echo "  $tag (8スレ, pid $!)"
done
