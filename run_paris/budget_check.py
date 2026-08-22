"""ダイナモランのエネルギー収支を dE/dt 込みで検証する。

Rempel (2006) 式 (20)-(22):

    dE_Omega/dt = Q_Lambda - Q_nu^Omega - Q_C - Q_L^Omega
    dE_M/dt     = Q_C - Q_nu^M - Q_B - Q_L^M
    dE_B/dt     = Q_L^Omega + Q_L^M - Q_eta

**左辺を落とさないこと。** 2026-08-23 に、スナップショット 1 点で
dE_B/dt = 0 と仮定した残差 (-0.005 〜 -0.016) を「未計上の項がある」と
誤って読んだ。実際には磁場がまだ成長中で dE_B/dt が入力の 10-25 パーセント
あった。時間微分を入れると残差は Q_Lambda 比で 1e-5 〜 1e-4 に落ちる
(原論文が明記している精度 0.001 より小さい)。

使い方::

    python run_paris/budget_check.py v2_kin v2_a125 v2_a250 v2_a500
"""
import sys, os
import numpy as np

YR = 3.156e7
QKEYS = ['Q_Lambda', 'Q_nu_Omega', 'Q_C', 'Q_L_Omega', 'Q_nu_M', 'Q_B',
         'Q_L_M', 'Q_eta']


def deriv(t, y, half=3, deg=4):
    """局所多項式当てはめによる時間微分.

    出力間隔は dt の再評価でわずかに揺らぐ (相対 3e-4) ので非一様に対応
    させる。2 次の中心差分との差は 3 パーセント以下 (実測) なので、
    残差が差分の打ち切り誤差でないことの確認にも使える。
    """
    out = np.full_like(y, np.nan)
    for i in range(half, len(y) - half):
        s = slice(i - half, i + half + 1)
        out[i] = np.polyfit(t[s] - t[i], y[s], deg)[-2]
    return out


def load(tag):
    d = np.load(f'results_rempel/{tag}/dynamo.npz')
    h = d['hist']
    if 'hist_cols' in d:
        cols = {k: h[:, i] for i, k in enumerate(d['hist_cols'])}
    else:
        # 2026-08-23 以前の記録には E_Omega / E_M が入っていない
        qk = list(d['qkeys'])
        cols = {'t': h[:, 0], 'E_B': h[:, 7]}
        cols.update({k: h[:, 10 + i] for i, k in enumerate(qk)})
    return cols


def report(tag):
    c = load(tag)
    t = c['t']
    n = len(t)
    k = max(4, n // 5)
    core = slice(k, -k)             # 両端 20 パーセントは微分が甘いので外す
    ql = np.abs(c['Q_Lambda'])
    print(f"=== {tag}: {n} 点, t = {t[0]/YR:.2f} - {t[-1]/YR:.2f} yr ===")
    print(f"{'収支式':>10}{'左辺 dE/dt':>26}{'残差 (dE/dt を落とす)':>24}"
          f"{'残差 (dE/dt 込み)':>22}")
    print(f"{'':10}{'平均':>13}{'最大':>13}{'平均':>12}{'最大':>12}"
          f"{'平均':>11}{'最大':>11}   (Q_Lambda 比)")

    budgets = [
        ('E_B', 'E_B',
         c['Q_L_Omega'] + c['Q_L_M'] - c['Q_eta']),
        ('E_Omega', 'E_Omega',
         c['Q_Lambda'] - c['Q_nu_Omega'] - c['Q_C'] - c['Q_L_Omega']),
        ('E_M', 'E_M',
         c['Q_C'] - c['Q_nu_M'] - c['Q_B'] - c['Q_L_M']),
    ]
    for name, key, rhs in budgets:
        if key not in c:
            print(f"{name:>10}   (記録に {key} がないので dE/dt を評価できない)")
            continue
        de = deriv(t, c[key])
        gap = rhs/ql                       # 左辺を落とした「残差」
        res = (rhs - de)/ql                # 本当の残差
        print(f"{name:>10}{np.mean(np.abs(de[core]/ql[core])):13.5f}"
              f"{np.max(np.abs(de[core]/ql[core])):13.5f}"
              f"{np.mean(np.abs(gap[core])):12.5f}{np.max(np.abs(gap[core])):12.5f}"
              f"{np.mean(np.abs(res[core])):11.6f}{np.max(np.abs(res[core])):11.6f}")
    print()


if __name__ == '__main__':
    tags = sys.argv[1:]
    if not tags:
        print(__doc__)
        sys.exit(1)
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for tag in tags:
        report(tag)
