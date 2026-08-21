#!/bin/bash
# 緩和の完走を待ってダイナモ 4 本を投入する。
# 人工拡散の弱い relax_cs010 を優先し、発散していたら relax_cs030 に落とす。
cd /scr/a000/c0234hotta/Repository/S2MFD || exit 1
echo "[chain] $(date '+%F %T') 待機開始"

wait_for () {   # tag -> 0:完走 1:発散/中断
  local tag=$1
  while true; do
    if grep -q "^\[$tag\] done" "results_rempel/$tag.log" 2>/dev/null; then return 0; fi
    if grep -q "発散" "results_rempel/$tag.log" 2>/dev/null; then return 1; fi
    if ! pgrep -f "uniform_rotation $tag\b" > /dev/null; then
      grep -q "^\[$tag\] done" "results_rempel/$tag.log" 2>/dev/null && return 0
      return 1
    fi
    sleep 120
  done
}

BASE=""
for tag in relax_cs010 relax_cs030; do
  echo "[chain] $(date '+%F %T') $tag を待つ"
  if wait_for "$tag"; then
    echo "[chain] $tag 完走"
    grep -E "表面差動回転|done" "results_rempel/$tag.log"
    BASE=$tag; break
  else
    echo "[chain] $tag は使えない (発散/中断)"; tail -3 "results_rempel/$tag.log"
  fi
done
[ -n "$BASE" ] || { echo "[chain] 使える緩和がない。中止"; exit 1; }

echo "[chain] $(date '+%F %T') $BASE を初期値にダイナモを投入"
CS=$( [ "$BASE" = relax_cs010 ] && echo 0.10 || echo 0.30 )
BASE=$BASE CS=$CS bash run_paris/launch_dynamo.sh
echo "[chain] $(date '+%F %T') 完了"
