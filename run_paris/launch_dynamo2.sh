#!/bin/bash
# ダイナモを修正版で続きから回す。
#  - Q_L^Omega の Omega_0 汚染を直した energy.py で診断を取り直す
#  - 磁場も final_state.npz から再開するので過渡なし
#  - さらに 60 年 (合計 120 年 = 約 7 サイクル) 回して統計を厚くする
cd /scr/a000/c0234hotta/Repository/S2MFD || exit 1
export MPLBACKEND=Agg PYTHONPATH=/scr/a000/c0234hotta/Repository/S2MFD
export OMP_NUM_THREADS=3 S2MFD_PARALLEL=3 S2MFD_PARFILE=parameters/rempel06_paper.py
PY=/scr/a000/c0234hotta/venv-paris/bin/python
launch () {  # src tag mode alpha
  local src=$1 tag=$2 mode=$3 a=$4
  setsid nohup $PY -u run_paris/dynamo7.py 108 72 0 60 "$tag" "$mode" \
      magnetic_buoyancy=1 sld_cs_factor=0.30 alpha0=$a \
      init=results_rempel/$src/final_state.npz \
      > results_rempel/$tag.log 2>&1 < /dev/null &
  echo "  $tag (続き, pid $!)"
}
launch p_kin  q_kin  kinematic 12.5
launch p_a125 q_a125 full      12.5
launch p_a250 q_a250 full      25.0
launch p_a500 q_a500 full      50.0
