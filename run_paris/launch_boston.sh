#!/bin/bash
# astana と paris が他ユーザで混んできたので boston に寄せる。
# 288x192 は退避した状態から再開、較正ラン 4 本は飽和状態を種に新規。
cd /scr/a000/c0234hotta/Repository/S2MFD || exit 1
export MPLBACKEND=Agg PYTHONPATH=/scr/a000/c0234hotta/Repository/S2MFD
export S2MFD_PARFILE=parameters/rempel06_paper.py
PY=/scr/a000/c0234hotta/venv-paris/bin/python
# 288x192 (12 スレッド) — 退避した t=3.6yr から残り 26 年
for cs in 0.30 0.10; do
  tag="s288x192_cs${cs/./}"
  OMP_NUM_THREADS=12 S2MFD_PARALLEL=12 \
  setsid nohup $PY -u run_paris/relax_scan.py 288 192 26 uniform_rotation "$tag" \
      sld_cs_factor=$cs init=results_rempel/move_$tag.npz \
      > results_rempel/$tag.log 2>&1 < /dev/null &
  echo "  $tag (12スレ, 種 t=3.6yr, pid $!)"
done
# 較正ラン (3 スレッド) — 0.27 を出すのに必要な cs を挟み込む
for cs in 0.40 0.50 0.70 1.00; do
  tag="cal_cs${cs/./}"
  OMP_NUM_THREADS=3 S2MFD_PARALLEL=3 \
  setsid nohup $PY -u run_paris/relax_scan.py 108 72 25 uniform_rotation "$tag" \
      sld_cs_factor=$cs init=results_rempel/m108x72_cs030/state.npz \
      > results_rempel/$tag.log 2>&1 < /dev/null &
  echo "  $tag (3スレ, pid $!)"
done
