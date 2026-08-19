"""フルスケール検証 (twin experiment + 観測データ推定) の結果を要約する。

使い方:
    python doc/dev_records/verification/summarize_full_runs.py \\
        --twin results/twin_full --obs results/datas_1944_1996
"""
import argparse
import json
import os

import numpy as np

from S2MFD.inference.observations import load_observations
from S2MFD.inference.problem import TimeFunction, U0_KEYS, S0_KEYS

YEAR_SECONDS = 365 * 24 * 3600


def fmt_pct(x):
    return f'{x*100:.2f}%'


def summarize_twin(path):
    f = os.path.join(path, 'twin_full_result.json')
    if not os.path.exists(f):
        print(f'[twin] 結果ファイルなし: {f}')
        return
    with open(f) as fh:
        r = json.load(fh)
    setup = r['setup']
    print('=' * 72)
    print('Twin experiment (合成データによるパラメタ回収試験)')
    print('=' * 72)
    print(f"  窓: {setup['start_year']:.2f} - {setup['end_year']:.2f} 年 "
          f"({setup['end_year']-setup['start_year']:.2f} 年)")
    print(f"  GA: 個体数 {setup['pop']}, 最大 {setup['generations']} 世代, "
          f"スピンアップ {setup['spinup_years']:.0f} 年, seed={setup['seed']}")
    print()
    print(f"{'':10} {'r_sunspot':>10} {'e_sunspot':>11} {'u0誤差':>9} {'s0誤差':>9} "
          f"{'fitness':>9} {'世代':>5}")
    for stage in ('step1', 'step2'):
        if stage not in r:
            continue
        s = r[stage]
        s0err = fmt_pct(s['s0_series_error']) if 's0_series_error' in s else '(定数)'
        print(f"{stage:10} {s['r_sunspot']:>10.4f} {s['e_sunspot']:>11.5f} "
              f"{fmt_pct(s['u0_series_error']):>9} {s0err:>9} "
              f"{s['fitness']:>9.5f} {s['generations_run']:>5}")
    print(f"{'論文表4.1':10} {0.995:>10.4f} {0.00514:>11.5f} {'2.87%':>9} {'5.25%':>9}")
    print()
    if 'step2' in r:
        print('  真値 vs 推定値:')
        for key in U0_KEYS:
            print(f"    {key:6} true={setup['true_u0'][key]:>8.2f}  "
                  f"est={r['step2']['stage1_u0'][key]:>8.2f}")
        for key in S0_KEYS:
            print(f"    {key:6} true={setup['true_s0'][key]:>8.2f}  "
                  f"est={r['step2']['params'][key]:>8.2f}")
        print()
        t = np.linspace(0, 1, 5)
        print('  時系列の比較 (窓を5等分した時点):')
        span = (setup['end_year'] - setup['start_year']) * YEAR_SECONDS
        f_u_true = TimeFunction(*[setup['true_u0'][k] for k in U0_KEYS], 0, span)
        f_u_est = TimeFunction(*[r['step2']['stage1_u0'][k] for k in U0_KEYS], 0, span)
        f_s_true = TimeFunction(*[setup['true_s0'][k] for k in S0_KEYS], 0, span)
        f_s_est = TimeFunction(*[r['step2']['params'][k] for k in S0_KEYS], 0, span)
        print(f"    {'年':>8} {'u0_true':>9} {'u0_est':>9} {'s0_true':>9} {'s0_est':>9}")
        for frac in t:
            yr = setup['start_year'] + frac * (setup['end_year'] - setup['start_year'])
            tt = frac * span
            print(f"    {yr:>8.1f} {f_u_true(tt):>9.1f} {f_u_est(tt):>9.1f} "
                  f"{f_s_true(tt):>9.2f} {f_s_est(tt):>9.2f}")
    print()


def summarize_obs(path):
    print('=' * 72)
    print('観測データ (SILSO) による推定')
    print('=' * 72)
    obs = load_observations('OBS')
    for stage in ('u0', 's0'):
        cand = [d for d in os.listdir(path) if d.startswith(f'data_{stage}_')] \
            if os.path.isdir(path) else []
        if not cand:
            print(f'[obs] {stage} の結果ディレクトリなし')
            continue
        f = os.path.join(path, cand[0], 'stage_result.json')
        if not os.path.exists(f):
            print(f'[obs] {stage}: 実行中または未完了')
            continue
        with open(f) as fh:
            r = json.load(fh)
        sc = r['rerun_scores']
        print(f"  Stage {stage}: fitness={r['fitness']:.5f} "
              f"({r['generations_run']} 世代{'・中断' if r['interrupted'] else ''})")
        print(f"    最終ラン: 相関 r={sc['cc']:.4f}, NMSE={sc['nmse']:.5f}, "
              f"EVA={sc['eva']:.5f}")
        print(f"    推定パラメタ: "
              + ', '.join(f'{k}={v:.2f}' for k, v in r['params'].items()))
        # 推定された振幅の変動幅
        keys = U0_KEYS if stage == 'u0' else S0_KEYS
        if all(k in r['params'] for k in keys):
            i0, i1 = r['config'].get('index_start'), r['config'].get('index_end')
            span = 52.05 * YEAR_SECONDS
            fn = TimeFunction(*[r['params'][k] for k in keys], 0, span)
            vals = np.array([fn(t) for t in np.linspace(0, span, 500)])
            mean = vals.mean()
            print(f"    振幅の変動: {vals.min():.1f} - {vals.max():.1f} "
                  f"(平均 {mean:.1f}, {100*(vals.min()-mean)/mean:+.1f}% "
                  f"〜 {100*(vals.max()-mean)/mean:+.1f}%)")
    print('  論文表4.4 (1944-1996): r_sunspot=0.873, e_sunspot=0.103')
    print('  論文の変動幅: u0 は -51.1% 〜 +29.0%, s0 は -69.1% 〜 +82.7%')
    print()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--twin', default='results/twin_full')
    p.add_argument('--obs', default='results/datas_1944_1996')
    a = p.parse_args()
    summarize_twin(a.twin)
    summarize_obs(a.obs)
