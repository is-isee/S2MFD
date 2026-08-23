#!/bin/bash
# 二重極限を詰めるために、cs が小さい系列を**解像度方向に**埋める。
#
# 2026-08-23 の収束表 (run_paris/convergence.py) で分かったこと:
#   - cs=0.30 は 108/144/216/288 の 4 点が飽和し、N -> 無限大 で 0.2688
#   - cs=0.10 は 144/216/288 の 3 点が飽和し 0.2757
#   - cs<=0.05 は 216x144 しか飽和した点がない -> 外挿できない
#
# そこで cs=0.05 を 108/144 でも飽和させ (どちらも未飽和のまま止まっていた)、
# 144x96 で cs=0.03/0.02 を新規に回す。108x72 は cs<=0.03 で発散するので出さない。
#
# 継続ランは t0= で時刻を引き継ぐ。これを渡さないと convergence.py の
# 「同じ (格子, cs) では最長のラン」が継続前の古いランを採ってしまう。
# 投入先はそのときの空き具合で決める (このスクリプトは host 非依存)。
# 2026-08-23 の実行時は astana が混んでいたので paris に出した。
# 投入前に ssh <host> "uptime; ps -eo user:20,nlwp,args" で
# load と**誰の何が走っているか**を見ること。
cd /scr/a000/c0234hotta/Repository/S2MFD || exit 1
export MPLBACKEND=Agg PYTHONPATH=/scr/a000/c0234hotta/Repository/S2MFD
export S2MFD_PARFILE=parameters/rempel06_paper.py
PY=/scr/a000/c0234hotta/venv-paris/bin/python
R=results_rempel

launch () {   # tag nx ny years threads init t0 cs
  local tag=$1 nx=$2 ny=$3 yr=$4 th=$5 init=$6 t0=$7 cs=$8
  OMP_NUM_THREADS=$th S2MFD_PARALLEL=$th \
  setsid nohup $PY -u run_paris/relax_scan.py $nx $ny $yr uniform_rotation "$tag" \
      sld_cs_factor=$cs init=$init t0=$t0 \
      > $R/$tag.log 2>&1 < /dev/null &
  echo "  $tag  ${nx}x${ny} cs=$cs  +${yr}yr (t0=${t0}yr)  ${th}スレ  pid $!"
}

# --- cs=0.05 を飽和させる (どちらも傾きが残っていた) ---
launch c108x72_cs005  108  72 60 3 $R/m108x72_cs005/state.npz 25 0.05
launch c144x96_cs005  144  96 60 4 $R/s144x96_cs005/state.npz 30 0.05
# --- cs=0.10 の 4 点目 (108x72 は 40 年で傾き +0.001 の境界) ---
launch c108x72_cs010  108  72 60 3 $R/m108x72_cs010/state.npz 40 0.10
# --- 144x96 で cs をさらに下げる (飽和した cs=0.05 の状態を種にする) ---
launch s144x96_cs003  144  96 50 4 $R/s144x96_cs005/state.npz  0 0.03
launch s144x96_cs002  144  96 50 4 $R/s144x96_cs005/state.npz  0 0.02
