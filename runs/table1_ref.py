"""磁場なし参照モデルを Rempel (2006) 表 1 **列 2** と突き合わせる。

表 1 列 2 (Reference):

    (Omega_eq - Omega_pole)/Omega_0   0.27
    Q_Lambda (F_sun)                  0.014
    Q_nu^Omega (Q_Lambda)             0.574
    Q_C (Q_Lambda)                    0.425
    Q_B (Q_Lambda)                    0.419
    Q_nu^M (Q_Lambda)                 0.005

表 1 の注:
    "the accuracy of the energy exchange terms is around 0.001, so the
     equilibrium relations Q_Lambda = Q_nu^Omega + Q_C + Q_L^Omega,
     Q_C = Q_nu^M + Q_B + Q_L^M, and Q_eta = Q_L^Omega + Q_L^M are only
     fulfilled within that error margin."

つまり **論文自身が 0.001 の精度**と明記している。本実装の残差もこの
基準で見る (`runs/budget_check.py` 参照)。

飽和したランだけを使う (convergence.py と同じ判定)。

使い方::

    python runs/table1_ref.py            # cs=0.30 の系列
    python runs/table1_ref.py 0.10       # cs を指定
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import convergence  # noqa: E402

L_SUN = 3.828e33          # [erg/s]
PAPER = {'DR': 0.27, 'QL_Fsun': 0.014, 'Qnu_Om': 0.574, 'QC': 0.425,
         'QB': 0.419, 'Qnu_M': 0.005}


def main(cs_target):
    tab = convergence.collect()
    rows = []
    for (nx, ny, cs), r in sorted(tab.items()):
        if abs(cs - cs_target) > 1e-12 or not r['sat']:
            continue
        h = np.load(f"results_rempel/{r['tag']}/history.npz")['hist']
        # 直近 1 割の平均を取る (最終点 1 個は揺らぐ)
        k = max(1, len(h)//10)
        dr, ql, qnu, qc, qnum, qb = (h[-k:, 1].mean(), h[-k:, 11].mean(),
                                     h[-k:, 12].mean(), h[-k:, 13].mean(),
                                     h[-k:, 14].mean(), h[-k:, 15].mean())
        rows.append(dict(tag=r['tag'], N=nx, t=r['t'], DR=dr,
                         QL_Fsun=ql/L_SUN, Qnu_Om=qnu/ql, QC=qc/ql,
                         QB=qb/ql, Qnu_M=qnum/ql,
                         # 式 (21) の残差 (定常なので dE_M/dt は落とす。
                         # 飽和したランなので妥当)
                         res21=(qc - qnum - qb)/ql))
    if not rows:
        print(f"cs={cs_target:g} に飽和したランがない")
        return
    keys = ['DR', 'QL_Fsun', 'Qnu_Om', 'QC', 'QB', 'Qnu_M']
    print(f"Rempel (2006) 表 1 列 2 との比較 (cs={cs_target:g}、飽和したランのみ)")
    print(f"{'格子':>8}{'t[yr]':>7}" + "".join(f"{k:>11}" for k in keys)
          + f"{'式(21)残差':>12}")
    print(f"{'論文':>8}{'':>7}" + "".join(f"{PAPER[k]:11.4f}" for k in keys)
          + f"{0.001:12.4f}")
    for r in rows:
        print(f"{r['N']:8d}{r['t']:7.0f}" + "".join(f"{r[k]:11.4f}" for k in keys)
              + f"{r['res21']:12.4f}")
    print(f"\n{'':8}{'':7}" + "".join(f"{'差':>11}" for k in keys))
    for r in rows:
        print(f"{r['N']:8d}{'':7}"
              + "".join(f"{r[k]-PAPER[k]:+11.4f}" for k in keys))


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 0.30)
