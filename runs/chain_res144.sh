#!/bin/bash
# res144_kin / res144_a125 の完走を待って、それぞれ 60 年まで延長する。
# 先に終わったほうから投入する (kin のほうが 6 年進んでいる)。
cd /scr/a000/c0234hotta/Repository/S2MFD || exit 1
echo "[chain144] $(date '+%F %T') 待機開始"

for tag in res144_kin res144_a125; do
  ( while true; do
      if grep -q "^\[$tag\] done" "results_rempel/$tag.log" 2>/dev/null; then
        echo "[chain144] $(date '+%F %T') $tag 完走 -> 延長を投入"
        TAG=$tag bash runs/launch_res144_extend.sh
        exit 0
      fi
      if grep -q "発散" "results_rempel/$tag.log" 2>/dev/null; then
        echo "[chain144] $(date '+%F %T') $tag は発散。延長しない"; exit 1
      fi
      if ! pgrep -f "dynamo7.py 144 96 0 42 $tag " > /dev/null; then
        echo "[chain144] $(date '+%F %T') $tag のプロセスが消えた。延長しない"; exit 1
      fi
      sleep 120
    done ) &
done
wait
echo "[chain144] $(date '+%F %T') 完了"
