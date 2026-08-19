"""全推定区間の結果を論文表4.4と並べて要約する。

使い方:
    python summarize_all_intervals.py [結果ルート ...]
"""
import glob
import json
import os
import sys

import numpy as np

# 論文表4.4 (Shimizu & Hotta 2026) の r_sunspot / e_sunspot
PAPER = {
    (1723, 1775): (0.927, 0.0551),
    (1775, 1833): (0.685, 0.264),
    (1833, 1889): (0.933, 0.0517),
    (1889, 1944): (0.935, 0.0675),
    (1944, 1996): (0.873, 0.103),
    (1996, 2024): (0.974, 0.0234),
}


def collect(roots):
    rows = {}
    for root in roots:
        tag = '' if os.path.basename(root) == 'results_full' else \
            os.path.basename(root).replace('results_full', '')
        for d in sorted(glob.glob(os.path.join(root, 'datas_*'))):
            name = os.path.basename(d)
            st, en = (int(x) for x in name.split('_')[1:3])
            entry = rows.setdefault((st, en, tag), {})
            for stage in ('u0', 's0'):
                hits = glob.glob(os.path.join(d, f'data_{stage}_*',
                                              'stage_result.json'))
                if hits:
                    with open(hits[0]) as f:
                        entry[stage] = json.load(f)
    return rows


def main():
    roots = sys.argv[1:] or [
        '/scr/a000/c0234hotta/Repository/S2MFD/results_full',
        '/scr/a000/c0234hotta/Repository/S2MFD/results_full_period',
    ]
    roots = [r for r in roots if os.path.isdir(r)]
    rows = collect(roots)

    print('=' * 78)
    print('観測データ (SILSO) 全区間の推定結果')
    print('=' * 78)
    hdr = (f"{'区間':>12} {'適応度':>7} {'Stage1 r':>9} {'Stage2 r':>9} "
           f"{'Stage2 NMSE':>11} {'論文 r':>7} {'世代(1/2)':>10}")
    print(hdr)
    print('-' * 78)
    ours = []
    for (st, en, tag) in sorted(rows):
        e = rows[(st, en, tag)]
        if 'u0' not in e:
            continue
        r1 = e['u0']['rerun_scores']['cc']
        g1 = e['u0']['generations_run']
        kind = e['u0']['config'].get('fitness_kind', 'standard')
        if 's0' in e:
            r2 = e['s0']['rerun_scores']['cc']
            n2 = e['s0']['rerun_scores']['nmse']
            g2 = str(e['s0']['generations_run'])
        else:
            r2, n2, g2 = float('nan'), float('nan'), '-'
        paper = PAPER.get((st, en), (float('nan'),))[0]
        label = f'{st}-{en}' + (tag if tag else '')
        print(f"{label:>12} {kind:>7} {r1:>9.4f} {r2:>9.4f} {n2:>11.4f} "
              f"{paper:>7.3f} {str(g1)+'/'+g2:>10}")
        if not tag and np.isfinite(r2) and np.isfinite(paper):
            ours.append((r2, paper))
    if ours:
        a = np.array(ours)
        print('-' * 78)
        print(f"  本実装 Stage2 の平均 r = {a[:, 0].mean():.3f} "
              f"(論文の平均 = {a[:, 1].mean():.3f}, {len(a)} 区間)")
    print()
    print('注: 適応度 standard = 論文式3.1 (0.9r - 0.1NMSE)、')
    print('    period = 周期の平均二乗誤差 (論文式4.2、ダルトン期用)')
    print('    世代数が少ない区間は目標適応度 0.80 に早期到達したもの')


if __name__ == '__main__':
    main()
