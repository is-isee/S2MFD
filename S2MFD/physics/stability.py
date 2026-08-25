"""線形安定性 (von Neumann) の判定。

この離散化は**中央差分 + SSP-RK2** なので、線形移流に対して無条件不安定
である:

.. math::
   z = -i\\nu\\sin\\theta,\\quad G = 1+z+\\tfrac12 z^2,\\quad
   |G|^2 = 1 + \\tfrac14\\nu^4\\sin^4\\theta > 1

CFL 数をいくら小さくしても増幅率は 1 を超える。**計算が成立しているのは
拡散のおかげ**であって「CFL 条件を満たしているから」ではない。したがって
拡散を弱める変更 (``sld_cs_factor`` を下げる等) は同時に安定性を削る。

SLD は波長 6 セル以上を「見ない」(設計どおり滑らかな解を削らない) ので、
**中間波数 (4-8 セル) を抑えられるのは物理拡散だけ**である。放射層では
:math:`\\nu_{\\rm dif}` が対流層値の 2% しかないため、そこが律速になる。

実測との関係
------------
非線形リミタのぶん、実際の安定限界は線形解析より甘い (2026-08-21 の実測で
約 3.5 倍)。``max_log_growth`` が正でも直ちに壊れるとは限らないが、
**正の設定は「非線形リミタの助けに頼っている」状態**であり、人工拡散を
弱めるほどその助けも減る。論文級の主張をするなら 0 以下で走らせること。

詳細: ``doc/dev_records/2026-08-21_cfl_von_neumann.md``
"""
import numpy as np

__all__ = ['sld_wavenumber_response', 'max_log_growth', 'neutral_cfl_safety']

_TH = np.linspace(1e-6, np.pi, 801)
_RESP_CACHE = {}


def sld_wavenumber_response(ep=2.0, fh=2.0, n=1024):
    """SLD の実効拡散の波数依存性を返す ``(theta, ratio)``.

    ``ratio`` は格子スケール (2 セル) での値を 1 とした比。
    正弦波 :math:`u=\\cos(k x)` にリミタを掛けて実効拡散係数を測る
    (``sin`` だと :math:`\\theta=\\pi` で恒等的にゼロになり誤る)。
    """
    key = (ep, fh, n)
    if key in _RESP_CACHE:
        return _TH, _RESP_CACHE[key]

    def one(theta):
        i = np.arange(n)
        u = np.cos(theta*i)
        a = u - np.roll(u, 1)
        b = np.roll(u, -1) - u
        c = 0.5*(a + b)
        A, B = ep*a, ep*b
        mx = np.maximum(np.maximum(A, B), c)
        mn = np.minimum(np.minimum(A, B), c)
        s_i = np.where(mx < 0, mx, 0.0) + np.where(mn > 0, mn, 0.0)
        ql = np.roll(u, 1) + 0.5*np.roll(s_i, 1)
        qr = u - 0.5*s_i
        dd = u - np.roll(u, 1)
        dd = np.where(np.abs(dd) > 1e-20, dd, 1e-20)
        ra = np.clip((qr - ql)/dd, None, 1.0)
        pp = np.where(ra <= 0.0, 0.0, np.maximum(0.0, 1.0 + fh*(ra - 1.0)))
        F = -0.5*pp*(qr - ql)
        dudt = -(np.roll(F, -1) - F)
        with np.errstate(divide='ignore', invalid='ignore'):
            val = -dudt/np.where(np.abs(u) > 1e-12, u, np.nan)
        v = np.nanmean(val[100:n-100])
        return v/(4*np.sin(0.5*theta)**2)/0.5 if theta > 1e-8 else 0.0

    resp = np.clip(np.nan_to_num(np.array([one(t) for t in _TH])), 0.0, None)
    _RESP_CACHE[key] = resp
    return _TH, resp


def max_log_growth(cfg, grid, strat, setup, dt):
    """全半径・全波数での :math:`\\max\\ln|G|` と最悪半径 ``(値, r/RSUN)``.

    0 以下なら線形安定。正なら非線形リミタの助けに頼っている。
    """
    th, resp = sld_wavenumber_response(getattr(cfg, 'sld_ep', 2.0),
                                       getattr(cfg, 'sld_fh', 2.0))
    m = grid.margin
    sl = slice(m, grid.ixg - m)
    rr = grid.rr[sl]
    dl = np.minimum(grid.drr[sl], rr*grid.dth)
    cs_eff = strat.cs_eff[sl]
    nu_t = setup.nu_dif[sl]
    cs_fac = getattr(cfg, 'sld_cs_factor', 0.3)
    sin_h2 = np.sin(0.5*th)**2
    sin_t = np.sin(th)
    worst, wr = -np.inf, np.nan
    for k in range(len(rr)):
        nu = cs_eff[k]*dt/dl[k]
        d = (nu_t[k] + 0.5*cs_fac*cs_eff[k]*dl[k]*resp)*dt/dl[k]**2
        z = -4.0*d*sin_h2 - 1j*nu*sin_t
        g = np.log(np.abs(1.0 + z + 0.5*z*z)).max()
        if g > worst:
            worst, wr = g, rr[k]/cfg.RSUN
    return float(worst), float(wr)


def neutral_cfl_safety(cfg, grid, strat, setup, dt_of_safety,
                       lo=0.005, hi=2.0, iters=40):
    """:math:`\\max\\ln|G|=0` になる ``cfl_safety`` を二分法で返す。

    ``dt_of_safety(S)`` は安全率 S に対する dt を返す呼び出し可能オブジェクト
    (通常 ``DynamicSolver.cfl_dt`` を cfg.cfl_safety を書き換えて呼ぶ)。
    線形安定に走らせたいときの上限になる。
    """
    if max_log_growth(cfg, grid, strat, setup, dt_of_safety(lo))[0] > 0.0:
        return float('nan')
    for _ in range(iters):
        mid = 0.5*(lo + hi)
        if max_log_growth(cfg, grid, strat, setup, dt_of_safety(mid))[0] <= 0.0:
            lo = mid
        else:
            hi = mid
    return lo
