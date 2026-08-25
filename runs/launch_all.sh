#!/bin/bash
# 拡散 CFL の修正後の再投入一式。
#   - 論文設定の緩和 40 年を 2 つ (cs_factor = 0.30 と 0.10)
#   - 人工拡散スキャン 25 年 (0.02 / 0.05 / 0.15 / 0.20)
#   - 解像度スキャン 25 年 (72x48 / 144x96 / 216x144)
cd /scr/a000/c0234hotta/Repository/S2MFD || exit 1
export MPLBACKEND=Agg PYTHONPATH=/scr/a000/c0234hotta/Repository/S2MFD
export OMP_NUM_THREADS=3 S2MFD_PARALLEL=3 S2MFD_PARFILE=parameters/rempel06_paper.py
PY=/scr/a000/c0234hotta/venv-paris/bin/python
run () {  # tag years nx ny extra...
  local tag=$1 yr=$2 nx=$3 ny=$4; shift 4
  setsid nohup $PY -u runs/relax_scan.py "$nx" "$ny" "$yr" uniform_rotation "$tag" \
      "$@" > results_rempel/$tag.log 2>&1 < /dev/null &
  echo "  $tag (pid $!)"
}
# 安全率は von Neumann の中立点 S0 以下に取る (線形安定に走らせる)。
#   cs <= 0.15 -> S0 = 0.314    (危険モードが 4-8 セルで SLD が効かない)
#   cs = 0.20  -> S0 = 0.394
#   cs = 0.30  -> S0 = 0.531
echo "本ラン (40 年):"
run relax_cs030 40 108 72 sld_cs_factor=0.30 cfl_safety=0.50
run relax_cs010 40 108 72 sld_cs_factor=0.10 cfl_safety=0.30
echo "人工拡散スキャン (25 年):"
run cs002 25 108 72 sld_cs_factor=0.02 cfl_safety=0.30
run cs005 25 108 72 sld_cs_factor=0.05 cfl_safety=0.30
run cs015 25 108 72 sld_cs_factor=0.15 cfl_safety=0.30
run cs020 25 108 72 sld_cs_factor=0.20 cfl_safety=0.38
echo "解像度スキャン (25 年, cs_factor=0.10):"
run res72x48   25 72  48 sld_cs_factor=0.10 cfl_safety=0.30
run res144x96  25 144 96 sld_cs_factor=0.10 cfl_safety=0.30
run res216x144 25 216 144 sld_cs_factor=0.10 cfl_safety=0.30
