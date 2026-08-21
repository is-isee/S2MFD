#!/bin/bash
# paper_relax の完走を待ってダイナモ 4 本を自動投入する。
cd /scr/a000/c0234hotta/Repository/S2MFD || exit 1
LOG=results_rempel/paper_relax.log
echo "[chain] $(date '+%F %T') 待機開始"
while true; do
  if grep -q "^\[paper_relax\] done" "$LOG" 2>/dev/null; then
    echo "[chain] $(date '+%F %T') 緩和完了を検出"
    grep -E "表面差動回転|done" "$LOG"
    break
  fi
  if ! pgrep -f "relax_scan.py 108 72 40 uniform_rotation paper_relax" > /dev/null; then
    echo "[chain] $(date '+%F %T') プロセスが消えたが done がない。中断とみなす"
    tail -5 "$LOG"; exit 1
  fi
  sleep 120
done
bash run_paris/launch_dynamo.sh
echo "[chain] $(date '+%F %T') ダイナモ投入完了"
