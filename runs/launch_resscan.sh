#!/bin/bash
# 解像度に対する差動回転の収束性。論文の解像度は 108x72 (北半球)。
# 15 年では定常に達しない (旧ラン r985t の収支残差 0.0211) ので 25 年回す。
cd /scr/a000/c0234hotta/Repository/S2MFD || exit 1
export MPLBACKEND=Agg PYTHONPATH=/scr/a000/c0234hotta/Repository/S2MFD
export OMP_NUM_THREADS=3 S2MFD_PARALLEL=3 S2MFD_PARFILE=parameters/rempel06_paper.py
PY=/scr/a000/c0234hotta/venv-paris/bin/python
launch () {
  setsid nohup $PY -u runs/relax_scan.py "$1" "$2" 25 uniform_rotation "res$1x$2" \
      > results_rempel/res$1x$2.log 2>&1 < /dev/null &
  echo "  res$1x$2 (pid $!)"
}
launch 72  48      # 論文の 2/3
launch 144 96      # 論文の 4/3
launch 216 144     # 論文の 2 倍
