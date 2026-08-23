"""Rempel (2006) 表 1 との比較 (比のドットプロット).

比は 1 をまたぐ**極性のある量**なので、1 を基準線にして左右で見せる。
色は系列の識別ではなく「どちらに外れているか」を担わないよう、単色 +
基準線で表す (過大・過小は位置が示す)。
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, 'run_paris')]
import style                                            # noqa: E402
import table1                                           # noqa: E402

ROWS = [('period [yr]', 'period', None),
        (r'max $B_\Phi$ [T]', 'Bph', 'Bph'),
        (r'max $B_r$ [T]', 'Br', 'Br'),
        (r'$\bar E_B$', 'EB', 'EB'),
        (r'max $|E_B-\bar E_B|/\bar E_B$', 'EBvar', 'EBvar'),
        (r'$Q_\Lambda/F_\odot$', 'QL_Fsun', 'QL'),
        (r'$Q_\nu^\Omega/Q_\Lambda$', 'Qnu', 'Qnu'),
        (r'$Q_C/Q_\Lambda$', 'QC', 'QC'),
        (r'$Q_L^\Omega/Q_\Lambda$', 'QLO', 'QLO'),
        (r'$Q_B/Q_\Lambda$', 'QB', 'QB'),
        (r'$Q_\nu^M/Q_\Lambda$', 'QnuM', 'QnuM'),
        (r'$Q_L^M/Q_\Lambda$', 'QLM', 'QLM'),
        (r'$Q_\eta/Q_\Lambda$', 'Qeta', 'Qeta'),
        (r'torsional osc. $90^\circ$', 't90', 't90'),
        (r'torsional osc. $60^\circ$', 't60', 't60'),
        (r'$(\Omega_{\rm eq}-\Omega_{\rm pole})/\Omega_0$', 'DR', 'DR')]
RUNS = [('v2_a125', 12.5), ('v2_a250', 25.0), ('v2_a500', 50.0)]
PERIOD_PAPER = 18.0


def main(outdir):
    style.apply()
    os.chdir(ROOT)
    cols = style.ordinal_blues(len(RUNS))
    fig, ax = plt.subplots(figsize=(8.4, 7.6))
    ax.axvline(1.0, color=style.REFERENCE, lw=1.6)
    ax.axvspan(0.9, 1.1, color=style.GRID, alpha=0.55, lw=0, zorder=0)

    ys = np.arange(len(ROWS))[::-1]
    for k, ((tag, a0), c) in enumerate(zip(RUNS, cols)):
        r = table1.analyse(tag)
        p = table1.PAPER[a0]
        off = (k - 1)*0.24
        xs, yy = [], []
        for i, (lab, rk, pk) in enumerate(ROWS):
            pv = PERIOD_PAPER if rk == 'period' else p.get(pk)
            v = r.get(rk)
            if v is None or pv in (None, 0):
                continue
            xs.append(v/pv); yy.append(ys[i] + off)
        ax.plot(xs, yy, 'o', color=c, markeredgecolor=style.SURFACE,
                markeredgewidth=1.2, ms=8, linestyle='none',
                label=f'$\\alpha_0$ = {a0/100:g} m s$^{{-1}}$', zorder=3)

    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in ROWS], fontsize=9)
    ax.set_ylim(-1.1, len(ROWS) - 0.4)
    ax.set_xlim(0.55, 1.55)
    ax.set_xlabel('this work / Rempel (2006) Table 1')
    ax.annotate('within $\\pm$10 %', xy=(1.0, -0.55), ha='center',
                fontsize=9, color=style.INK2)
    ax.legend(loc='lower right', ncol=1)
    ax.grid(axis='y', color=style.GRID, lw=0.6)
    ax.set_title('Nonkinematic dynamo vs Rempel (2006) Table 1\n'
                 '$108\\times72$,  $c_s$ = 0.30,  60 yr',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    out = os.path.join(outdir, 'table1_comparison.png')
    fig.savefig(out)
    print('wrote', out)


if __name__ == '__main__':
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'out')
    os.makedirs(d, exist_ok=True)
    main(d)
