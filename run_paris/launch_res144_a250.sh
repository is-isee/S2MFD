#!/bin/bash
# alpha0 = 25 cm/s を 144x96 で 60 年。
#
# 目的 (2026-08-24): トーショナル振動の極 / 緯度 60 度の比が論文と合わない
# のは a250 の 1 点だけ (1.25 対 1.67)。a125 と a500 は 1 割で合っている。
# 解像度で動くかどうかで、極の扱いの問題か測定誤差かが決まる。
# 詳細は doc/dev_records/2026-08-24_torsional_pole.md
#
# 延長を挟まず 60 年を一気に走らせる (42 年では飽和していないことが
# 分かっているため)。表面 Omega_1 の全緯度記録 (torsional) が入る最初の
# ランでもある。
cd /scr/a000/c0234hotta/Repository/S2MFD || exit 1
TAG=${TAG:-res144_a250}
YEARS=${YEARS:-60}
NTH=${NTH:-6}
if pgrep -f "dynamo7.py 144 96 0 .* $TAG " > /dev/null; then
  echo "  $TAG はすでに走っている。投入しない"; exit 1
fi
if [ -e "results_rempel/$TAG.log" ]; then
  echo "  results_rempel/$TAG.log がすでにある。投入しない"; exit 1
fi
export MPLBACKEND=Agg PYTHONPATH=/scr/a000/c0234hotta/Repository/S2MFD
export S2MFD_PARFILE=parameters/rempel06_paper.py
PY=/scr/a000/c0234hotta/venv-paris/bin/python
OMP_NUM_THREADS=$NTH S2MFD_PARALLEL=$NTH \
setsid nohup nice -n 5 $PY -u run_paris/dynamo7.py 144 96 0 "$YEARS" "$TAG" full \
    magnetic_buoyancy=1 bseed=3000 sld_cs_factor=0.30 alpha0=25.0 \
    init=results_rempel/s144x96_cs030/state.npz \
    > "results_rempel/$TAG.log" 2>&1 < /dev/null &
echo "  $TAG (144x96 alpha0=25, ${YEARS}年, ${NTH}スレ, pid $!)"
