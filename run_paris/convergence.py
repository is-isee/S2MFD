"""解像度 x 人工拡散の二重極限を、**飽和したランだけ**で見る。

`status.py` の要約表は ``m*`` で始まるタグしか拾わないので、後から投入した
``s216x144_*`` / ``s288x192_*`` を無視して古い短いランの値を表示していた
(2026-08-23 に発見。216x144 cs=0.30 を「+0.2121@9y」と出していたが、実際は
`s216x144_cs030` が 28 年走って +0.2777)。ここでは接頭辞によらず
(格子, cs) ごとに**最も長く走ったラン**を採る。

飽和の判定は dDR/dt。過渡の途中の値で収束先を語らないための道具である
(`doc/dev_records/2026-08-23_mistakes.md` の失敗パターン B)。

使い方::

    python run_paris/convergence.py             # 表
    python run_paris/convergence.py --extrap    # cs 固定で N -> 無限大 に外挿
    python run_paris/convergence.py --fit       # 2 次元当てはめ (筋が悪い。下記)

2 次元当てはめ ``DR = DR_inf + a/N^2 + b*cs/N`` は残差 rms 0.012 で、しかも
残差が構造的だった (2026-08-23)。**cs を固定して N で外挿する方が素直**で、
cs=0.30 の 4 点は p=2.41 で rms 0.0007 に収まる。``--fit`` は記録として
残してあるだけ。
"""
import glob
import os
import re
import sys

import numpy as np

YR = 3.156e7
#: |dDR/dt| がこれ未満なら飽和とみなす [1/yr]
SATURATED = 1.0e-3
#: 傾きを測る窓 [yr]
WINDOW = 5.0
#: **その cs で**これだけの年数を回っていないと飽和と判定しない [yr]
#
# 傾きだけで判定すると、別の cs の緩和済み状態を種にして数年しか回して
# いないランを「飽和」と誤判定する (2026-08-23 に s288x192_cs005 が 3 年で
# 飽和判定になった)。緩和は 216x144 級で 20-30 年かかる。
#
# 「その cs での緩和時間」は hist の **絶対時刻 t[-1]** で測る。
# `relax_scan.py` の `t0=` は**同じ cs の継続ランにだけ渡す**規約なので、
# 継続ランでは t[-1] が通算の緩和時間になり、cs を変えて種にしただけの
# ランでは t0=0 のままなので t[-1] がその cs での緩和時間になる。
MIN_SPAN = 10.0

TAG = re.compile(r'^[a-z]*?(\d+)x(\d+)_cs(\d+)$')


def _cs_from_tag(s):
    """'030' -> 0.30, '005' -> 0.05, '002' -> 0.02."""
    return int(s)/1000.0*10.0 if len(s) == 3 else float(s)/100.0


def saturate_fit(t, dr, frac=0.6):
    """DR(t) = DR_inf - A exp(-t/tau) を当てはめて、飽和値と残りを返す。

    傾きの閾値だけで判定すると、**人工拡散が弱いランは緩和が遅すぎて**
    未飽和のまま通ってしまう (2026-08-23: c108x72_cs010 が 100 年走っても
    +0.0008/yr で上がり続けていたのに「飽和」と判定されていた)。
    緩和曲線を外挿して**残りいくら動くか**を出す方が使える。

    Returns
    -------
    (DR_inf, tau, gap) : float
        ``gap = DR_inf - DR(最新)``。当てはまらなければ ``(nan, nan, nan)``。
    """
    from scipy.optimize import least_squares
    k = int(len(t)*(1.0 - frac))
    tt, yy = t[k:], dr[k:]
    if len(tt) < 12:
        return float('nan'), float('nan'), float('nan')
    span = tt[-1] - tt[0]
    sign = 1.0 if yy[-1] >= yy[0] else -1.0

    def res(q):
        return q[0] - q[1]*np.exp(-(tt - tt[0])/max(q[2], 1e-3)) - yy

    try:
        r = least_squares(res, [yy[-1] + sign*0.01, sign*0.01, span],
                          bounds=([-1.0, -1.0, 0.5*span/10.0],
                                  [1.0, 1.0, 50.0*span]))
    except Exception:
        return float('nan'), float('nan'), float('nan')
    # tau が窓より極端に長いと外挿が効かない (ほぼ直線の当てはめ)
    if not np.isfinite(r.x[0]) or r.x[2] > 20.0*span:
        return float('nan'), float('nan'), float('nan')
    return float(r.x[0]), float(r.x[2]), float(r.x[0] - yy[-1])


def collect():
    """(nx, ny, cs) -> dict の辞書。同じ組は最長のランを採る。"""
    out = {}
    for p in sorted(glob.glob('results_rempel/*/history.npz')):
        tag = os.path.basename(os.path.dirname(p))
        m = TAG.match(tag)
        if not m:
            continue
        try:
            h = np.load(p)['hist']
        except Exception:
            continue
        if len(h) < 10:
            continue
        t = h[:, 0]/YR
        dr = h[:, 1]
        # 直近 WINDOW 年の傾き
        sel = t > t[-1] - WINDOW
        slope = (np.polyfit(t[sel], dr[sel], 1)[0] if sel.sum() >= 3
                 else float('nan'))
        key = (int(m[1]), int(m[2]), _cs_from_tag(m[3]))
        # t0= の規約 (上の MIN_SPAN の説明) により、絶対時刻がそのまま
        # 「その cs での緩和時間」になる
        span = t[-1]
        dinf, tau, gap = saturate_fit(t, dr)
        # 緩和曲線の外挿で「残りどれだけ動くか」を見る。傾きだけだと
        # 緩和が遅いランを取りこぼす (saturate_fit の docstring 参照)。
        settled = (abs(gap) < 0.002) if np.isfinite(gap) else (
            abs(slope) < SATURATED)
        rec = dict(tag=tag, t=t[-1], DR=dr[-1], slope=slope, span=span,
                   DR_inf=dinf, tau=tau, gap=gap,
                   sat=(settled and abs(slope) < SATURATED
                        and span >= MIN_SPAN),
                   short=span < MIN_SPAN, n=len(h))
        if key not in out or rec['t'] > out[key]['t']:
            out[key] = rec
    return out


def table(tab):
    css = sorted({k[2] for k in tab}, reverse=True)
    reso = sorted({(k[0], k[1]) for k in tab})
    print(f"DR (飽和 = |dDR/dt| < {SATURATED:g}/yr かつ自身の走行 "
          f">= {MIN_SPAN:g} 年)。* は未飽和、? は走行が短くて判定保留")
    print(f"{'格子':>10}" + "".join(f"{f'cs={c:g}':>18}" for c in css))
    for nx, ny in reso:
        line = f"{nx}x{ny:<6}"
        for c in css:
            r = tab.get((nx, ny, c))
            if r is None:
                line += f"{'-':>18}"
            else:
                mark = ' ' if r['sat'] else ('?' if r['short'] else '*')
                line += f"{f'{r[chr(68)+chr(82)]:+.4f}{mark}@{r[chr(116)]:.0f}y':>18}"
        print(line)
    print()
    print(f"{'ラン':22}{'t[yr]':>8}{'DR':>9}{'dDR/dt':>10}"
          f"{'外挿 DR_inf':>12}{'残り':>9}{'tau[yr]':>9}  判定")
    for k in sorted(tab):
        r = tab[k]
        v = ('飽和' if r['sat']
             else ('判定保留 (走行 %.0f 年)' % r['span'] if r['short']
                   else '**未飽和**'))
        di = f"{r['DR_inf']:+12.4f}" if np.isfinite(r['DR_inf']) else f"{'-':>12}"
        gp = f"{r['gap']:+9.4f}" if np.isfinite(r['gap']) else f"{'-':>9}"
        ta = f"{r['tau']:9.1f}" if np.isfinite(r['tau']) else f"{'-':>9}"
        print(f"{r['tag']:22}{r['t']:8.1f}{r['DR']:+9.4f}{r['slope']:+10.5f}"
              f"{di}{gp}{ta}  {v}")


def fit(tab):
    """飽和した点だけで DR(N, cs) = DR_inf + a/N^2 + b*cs/N を当てはめる。

    SLD の人工拡散係数は kappa ~ 0.5 * cs * c * dx なので **cs/N に比例**し、
    打ち切り誤差は 1/N^2 に比例する、という想定。**当てはめであって
    証明ではない**ので、残差と、点を落としたときの安定性を必ず見ること。
    """
    pts = [(k[0], k[2], v['DR']) for k, v in tab.items() if v['sat']]
    if len(pts) < 4:
        print(f"\n飽和した点が {len(pts)} 個しかないので当てはめない")
        return
    N = np.array([p[0] for p in pts], float)
    cs = np.array([p[1] for p in pts], float)
    dr = np.array([p[2] for p in pts], float)
    A = np.column_stack([np.ones_like(N), 1.0/N**2, cs/N])
    coef, *_ = np.linalg.lstsq(A, dr, rcond=None)
    pred = A @ coef
    print(f"\n=== 飽和した {len(pts)} 点で DR = DR_inf + a/N^2 + b*cs/N ===")
    print(f"  DR_inf = {coef[0]:+.4f}   a = {coef[1]:+.4g}   b = {coef[2]:+.4g}")
    print(f"  残差 rms = {np.sqrt(np.mean((dr-pred)**2)):.5f} "
          f"(最大 {np.abs(dr-pred).max():.5f})")
    print(f"  Rempel (2006) 表 1 列 2 = 0.27")
    print(f"\n{'N':>6}{'cs':>7}{'DR 実測':>10}{'当てはめ':>10}{'差':>9}")
    for (n, c, d), p in sorted(zip(pts, pred)):
        print(f"{n:6.0f}{c:7.3g}{d:+10.4f}{p:+10.4f}{d-p:+9.4f}")
    # 1 点抜きの安定性
    if len(pts) > 4:
        infs = []
        for i in range(len(pts)):
            m = np.ones(len(pts), bool); m[i] = False
            c2, *_ = np.linalg.lstsq(A[m], dr[m], rcond=None)
            infs.append(c2[0])
        print(f"\n  1 点抜きの DR_inf: {min(infs):+.4f} 〜 {max(infs):+.4f} "
              f"(幅 {max(infs)-min(infs):.4f})")


def extrapolate(tab, min_points=3):
    """cs を固定して DR = DR_inf + a N^-p を当てはめ、N -> 無限大 を見る。

    **飽和した点だけ**を使う。p も当てはめるので 3 点では自由度がゼロに
    なり、残差が出ない (良い当てはめに見えるだけ)。4 点以上の系列だけを
    信用すること。
    """
    from scipy.optimize import least_squares
    series = {}
    for (nx, ny, cs), r in tab.items():
        if r['sat']:
            series.setdefault(cs, []).append((nx, r['DR']))
    print(f"\n=== cs 固定で N -> 無限大 に外挿 (飽和した点のみ) ===")
    print(f"{'cs':>6}{'点':>4}{'N':>26}{'DR_inf':>10}{'p':>7}{'rms':>9}")
    for cs in sorted(series, reverse=True):
        pts = sorted(series[cs])
        if len(pts) < min_points:
            ns = ','.join(str(n) for n, _ in pts)
            print(f"{cs:6.2f}{len(pts):4d}{ns:>26}{'(点が足りない)':>26}")
            continue
        N = np.array([q[0] for q in pts], float)
        y = np.array([q[1] for q in pts])

        def res(q):
            return q[0] + q[1]*N**(-q[2]) - y

        r = least_squares(res, [0.27, 100.0, 2.0],
                          bounds=([-1, -1e6, 0.5], [1, 1e6, 6]))
        rms = np.sqrt(np.mean(r.fun**2))
        ns = ','.join(str(int(n)) for n in N)
        flag = '' if len(pts) > 3 else '  (3 点なので残差はゼロになる)'
        print(f"{cs:6.2f}{len(pts):4d}{ns:>26}{r.x[0]:+10.4f}{r.x[2]:7.2f}"
              f"{rms:9.5f}{flag}")
        # 最も粗い点を落としても同じ値に来るか (漸近領域に入っているかの目安)
        if len(pts) > 3:
            N2, y2 = N[1:], y[1:]

            def res2(q):
                return q[0] + q[1]*N2**(-q[2]) - y2

            r2 = least_squares(res2, [0.27, 100.0, 2.0],
                               bounds=([-1, -1e6, 0.5], [1, 1e6, 6]))
            print(f"{'':10}{'最粗を落とす':>26}{r2.x[0]:+10.4f}{r2.x[2]:7.2f}"
                  f"{'':9}  (差 {r2.x[0]-r.x[0]:+.4f})")
    print(f"  Rempel (2006) 表 1 列 2 = 0.27")
    # 固定 N での cs 依存の幅 (二重極限が交換するかの目安)
    print(f"\n=== 固定 N での cs 依存の幅 (cs=0.10 と 0.30 の差) ===")
    for nx in sorted({k[0] for k in tab}):
        # ny は格子ごとに決まるので、キーから引く (2/3 を仮定しない)
        def pick(cs):
            for (kx, ky, kc), v in tab.items():
                if kx == nx and abs(kc - cs) < 1e-12 and v['sat']:
                    return v
            return None

        a, b = pick(0.10), pick(0.30)
        if a and b:
            print(f"  N={nx:4d}  {a['DR']-b['DR']:+.4f}")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    tab = collect()
    table(tab)
    if '--extrap' in sys.argv:
        extrapolate(tab)
    if '--fit' in sys.argv:
        fit(tab)
