#!/bin/bash
# 論文どおりの下部境界 (B_Phi = 0) が 108x72 で発散したのは、深部の小さい
# eta (1e5 m^2/s) が作る抵抗層が 1 セルに入らないためか、を調べる。
#
#   格子      底の dr    抵抗層 sqrt(eta*1yr)   層/dr
#   108x72   0.00386R      0.00255R            0.66   <- 入らない
#   216x144  0.00195R      0.00255R            1.31
#   288x192  0.00147R      0.00255R            1.74
#
# 216x144 で安定なら「論文の格子では解けない境界層だった」と言える。
# 発散するなら本実装のスキームと相性が悪いということ。
#
# 対照は同じ 216x144 の 'vertical'。運動学的なので流体は動かない。
cd /scr/a000/c0234hotta/Repository/S2MFD || exit 1
export MPLBACKEND=Agg PYTHONPATH=/scr/a000/c0234hotta/Repository/S2MFD
export S2MFD_PARFILE=parameters/rempel06_paper.py
PY=/scr/a000/c0234hotta/venv-paris/bin/python
for bc in R06 vertical; do
  tag="bc216_${bc}"
  OMP_NUM_THREADS=8 S2MFD_PARALLEL=8 \
  setsid nohup $PY -u run_paris/dynamo7.py 216 144 0 14 "$tag" kinematic \
      magnetic_buoyancy=1 bseed=3000 sld_cs_factor=0.30 alpha0=12.5 \
      boundary_condition_type=$bc \
      init=results_rempel/s216x144_cs030/state.npz \
      > results_rempel/$tag.log 2>&1 < /dev/null &
  echo "  $tag (8スレ, pid $!)"
done
