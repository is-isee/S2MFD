#!/bin/bash
# res144_kin / res144_a125 を通算 60 年まで延長する。
#
# 2026-08-24: 同じ時間窓 (19-39 年) で比べると、解像度を 108x72 から
# 144x96 に上げても周期はほとんど動かない (16.9 対 17.0 年)。一方
# p_kin / v2_a125 の反転間隔は 16.8 -> 18.0 と**時間とともに伸び**、
# max|B_phi| と E_B も 55 年まで単調に増える。**42 年では飽和していない。**
# 論文の表 1 と比べるには 108x72 の 60 年ランと同じ長さが要る。
#
# 使い方:  TAG=res144_kin  bash runs/launch_res144_extend.sh
#          TAG=res144_a125 bash runs/launch_res144_extend.sh
# 42 年の final_state.npz から 18 年ぶん再開する (時刻も引き継がれる)。
TAG=${TAG:-res144_kin}
case "$TAG" in
  res144_kin)  MODE=kinematic ;;
  res144_a125) MODE=full ;;
  *) echo "TAG は res144_kin か res144_a125"; exit 1 ;;
esac
YEARS=${YEARS:-18}
cd /scr/a000/c0234hotta/Repository/S2MFD || exit 1
INIT=results_rempel/$TAG/final_state.npz
[ -f "$INIT" ] || { echo "$INIT がない ($TAG はまだ完走していない)"; exit 1; }
# **二重投入を防ぐ。** 2026-08-24 に、利用者が手で投入した 88 秒後に
# chain_res144.sh が同じものを投入し、同じ results_rempel/${TAG}_b/ へ
# 2 プロセスが書いた (計算は決定論的なので中身は同じだったが、np.savez
# が競合すれば壊れうるし、コアも二重に食う)。
if pgrep -f "dynamo7.py 144 96 0 .* ${TAG}_b " > /dev/null; then
  echo "  ${TAG}_b はすでに走っている。投入しない"; exit 1
fi
if [ -e "results_rempel/${TAG}_b.log" ]; then
  echo "  results_rempel/${TAG}_b.log がすでにある。投入しない"
  echo "  (やり直すなら results_rempel/${TAG}_b{,.log} を消してから)"; exit 1
fi
export MPLBACKEND=Agg PYTHONPATH=/scr/a000/c0234hotta/Repository/S2MFD
export S2MFD_PARFILE=parameters/rempel06_paper.py
PY=/scr/a000/c0234hotta/venv-paris/bin/python
OMP_NUM_THREADS=5 S2MFD_PARALLEL=5 \
setsid nohup $PY -u runs/dynamo7.py 144 96 0 "$YEARS" "${TAG}_b" "$MODE" \
    magnetic_buoyancy=1 sld_cs_factor=0.30 alpha0=12.5 \
    init="$INIT" \
    > "results_rempel/${TAG}_b.log" 2>&1 < /dev/null &
echo "  ${TAG}_b (144x96 $MODE, +${YEARS}年 = 通算 60年, 5スレ, pid $!)"
