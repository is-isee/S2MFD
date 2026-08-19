#!/bin/bash
cd /scr/a000/c0234hotta/Repository/S2MFD || exit 1
/scr/a000/c0234hotta/venv-paris/bin/python - <<'PY'
import json
def load(p):
    with open(p) as f: return json.load(f)
base='/scr/a000/c0234hotta/Repository/S2MFD/results_paris/datas_1944_1996'
opt='/scr/a000/c0234hotta/bench-wt/results_opt/datas_1944_1996'
print(f"{'':8} {'パラメタ':8} {'原実装 (main)':>24} {'最適化版':>24} {'一致':>6}")
allsame=True
for stage in ('u0','s0'):
    a=load(f'{base}/data_{stage}_1944_1996/stage_result.json')
    b=load(f'{opt}/data_{stage}_1944_1996/stage_result.json')
    for k in a['params']:
        same = a['params'][k]==b['params'][k]
        allsame &= same
        print(f"  {stage:6} {k:8} {a['params'][k]:>24.15g} {b['params'][k]:>24.15g} {'○' if same else '×':>6}")
    for k in ('fitness','generations_run'):
        same = a[k]==b[k]; allsame &= same
        print(f"  {stage:6} {k:8} {a[k]:>24} {b[k]:>24} {'○' if same else '×':>6}")
    for k in ('cc','nmse','eva'):
        d=abs(a['rerun_scores'][k]-b['rerun_scores'][k])
        print(f"  {stage:6} {k:8} {a['rerun_scores'][k]:>24.15g} {b['rerun_scores'][k]:>24.15g} {'差 %.1e'%d:>10}")
print()
print('全パラメタ・適応度・世代数が完全一致' if allsame else '一部不一致あり')
PY
