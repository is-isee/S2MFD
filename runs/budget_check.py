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

**E_Omega の左辺は差動回転のエネルギー (Omega_1 だけ) で評価すること。**
E_Omega = int (1/2) rho0 varpi^2 (Omega_0 + Omega_1)^2 の微分には
Omega_0 * dL/dt に比例する項が入る。全角運動量保存から解析的に厳密ゼロ
だが離散では消えず、Omega_0 が大きいので本物の信号を飲み込む
(実測 0.19 対 0.015、10 倍以上)。

使い方::

    python runs/budget_check.py v2_kin v2_a125          # ダイナモ
    python runs/budget_check.py --relax m108x72_cs030   # 磁場なし
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
        # 2026-08-23 以前の記録には E_Omega / E_M が入っていない。
        # 古い記録は 16 列 = 診断 8 + QKEYS 8 で、QKEYS は 10 でなく 8 から。
        qk = [str(k) for k in d['qkeys']]
        off = h.shape[1] - len(qk)
        cols = {'t': h[:, 0], 'E_B': h[:, 7]}
        cols.update({k: h[:, off + i] for i, k in enumerate(qk)})
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


# relax_scan の hist の列
RELAX_COLS = ['t', 'DR', 'om1_max', 'vr', 'vth', 'se1_max', 'L_res', 'M_res',
              'E_Omega', 'E_M', 'E_DR', 'Q_Lambda', 'Q_nu_Omega', 'Q_C',
              'Q_nu_M', 'Q_B']


def report_relax(tag):
    """磁場なしラン (relax_scan) の式 (20) (21) を検証する。"""
    h = np.load(f'results_rempel/{tag}/history.npz')['hist']
    if h.shape[1] < len(RELAX_COLS):
        print(f"{tag}: 列が {h.shape[1]} 本しかない (E_Omega 以降がない)")
        return
    c = {k: h[:, i] for i, k in enumerate(RELAX_COLS)}
    t = c['t']
    k = max(4, len(t)//5)
    core = slice(k, -k)
    a = np.abs(c['Q_Lambda'])
    rhs_om = c['Q_Lambda'] - c['Q_nu_Omega'] - c['Q_C']
    rhs_m = c['Q_C'] - c['Q_nu_M'] - c['Q_B']
    r_om = (rhs_om - deriv(t, c['E_Omega']))/a       # 誤り: Omega_0 が入る
    r_dr = (rhs_om - deriv(t, c['E_DR']))/a          # 正しい
    r_m = (rhs_m - deriv(t, c['E_M']))/a
    print(f"{tag:22}{len(t):5d}{t[-1]/YR:8.1f}"
          f"{np.mean(np.abs(r_om[core])):14.5f}{np.mean(np.abs(r_dr[core])):14.5f}"
          f"{np.mean(np.abs(r_m[core])):12.5f}")


if __name__ == '__main__':
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if args[0] == '--relax':
        print(f"{'ラン':22}{'n':>5}{'t[yr]':>8}"
              f"{'残差 (E_Omega)':>14}{'残差 (E_DR)':>14}{'残差 (E_M)':>12}"
              f"   (Q_Lambda 比)")
        for tag in args[1:]:
            report_relax(tag)
    else:
        for tag in args:
            report(tag)
