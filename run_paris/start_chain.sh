#!/bin/bash
cd /scr/a000/c0234hotta/Repository/S2MFD || exit 1
setsid nohup bash run_paris/chain.sh > results_rempel/chain.log 2>&1 < /dev/null &
echo "chain pid=$!"
