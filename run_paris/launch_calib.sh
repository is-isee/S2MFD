#!/bin/bash
# 「Rempel の 0.27 を論文の解像度 108x72 で再現するのに必要な人工拡散」を逆算する。
# 得られた cs_factor が、彼の MacCormack (交互風上/風下差分) の数値散逸の
# 我々のスキームでの等価量になる (散逸する波数帯が違うので近似ではある)。
#
# cs=0.30 で DR=0.322 なので、0.27 にはもっと強い拡散が要る。
# 種は 108x72 cs=0.30 の飽和状態 (t=39.9yr) を使い、過渡を飛ばす。
cd /scr/a000/c0234hotta/Repository/S2MFD || exit 1
export MPLBACKEND=Agg PYTHONPATH=/scr/a000/c0234hotta/Repository/S2MFD
export OMP_NUM_THREADS=3 S2MFD_PARALLEL=3 S2MFD_PARFILE=parameters/rempel06_paper.py
PY=/scr/a000/c0234hotta/venv-paris/bin/python
SEED=results_rempel/m108x72_cs030/state.npz
for cs in 0.40 0.50 0.70 1.00; do
  tag="cal_cs${cs/./}"
  setsid nohup $PY -u run_paris/relax_scan.py 108 72 25 uniform_rotation "$tag" \
      sld_cs_factor=$cs init=$SEED > results_rempel/$tag.log 2>&1 < /dev/null &
  echo "  $tag (cs=$cs, pid $!)"
done
