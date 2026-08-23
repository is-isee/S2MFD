"""(sld_cs_factor, cfl_safety) 平面での線形安定境界を出す。

2026-08-21 の解析 (doc/dev_records/2026-08-21_cfl_von_neumann.md) は
sld_cs_factor = 0.3 に固定して安全率 S だけを振っていた。人工拡散を
下げると安定性も同時に削れるので、cs_factor ごとに許される S を出す。

構造 (前回の解析で判明していること):
* 中央差分 + SSP-RK2 は移流に対して無条件不安定 (|G|^2 = 1 + s^4/4)。
  安定なのは拡散のおかげ。
* SLD は 6 セル以上の波長を「見ない」(Phi_eff = 0)。したがって中間波数
  (4-8 セル) を抑えるのは**物理拡散だけ**。放射層では nu_dif が 2% フロア
  しかないので、そこが律速になる。
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
np.seterr(all='ignore')
import S2MFD
from S2MFD.stratification import Stratification
from S2MFD.physics.dynamic import DynamicSolver

TH = np.linspace(1e-6, np.pi, 801)

def sld_response(theta, ep=2.0, fh=2.0, n=1024):
    """波数 theta の正弦波に対する SLD の実効拡散 (格子スケールを 1 とする比)。"""
    i = np.arange(n); u = np.cos(theta*i)
    def mm3(a, b):
        c = 0.5*(a+b); A = ep*a; B = ep*b
        mx = np.maximum(np.maximum(A, B), c); mn = np.minimum(np.minimum(A, B), c)
        return np.where(mx < 0, mx, 0.0) + np.where(mn > 0, mn, 0.0)
    d0 = np.roll(u, -1) - u; dm = u - np.roll(u, 1)
    s_i = mm3(dm, d0); s_im = np.roll(s_i, 1)
    ql = np.roll(u, 1) + 0.5*s_im; qr = u - 0.5*s_i
    dd = u - np.roll(u, 1); dd = np.where(np.abs(dd) > 1e-20, dd, 1e-20)
    ra = np.clip((qr - ql)/dd, None, 1.0)
    pp = np.where(ra <= 0.0, 0.0, np.maximum(0.0, 1.0 + fh*(ra - 1.0)))
    F = -0.5*pp*(qr - ql)
    dudt = -(np.roll(F, -1) - F)
    with np.errstate(divide='ignore', invalid='ignore'):
        val = -dudt/np.where(np.abs(u) > 1e-12, u, np.nan)
    v = np.nanmean(val[100:n-100])
    return v/(4*np.sin(0.5*theta)**2)/0.5 if theta > 1e-8 else 0.0

RESP = np.clip(np.nan_to_num(np.array([sld_response(t) for t in TH])), 0, None)

def build(cs):
    cfg = S2MFD.build_cfg('parameters/rempel06_paper.py', sld_cs_factor=cs)
    grid = S2MFD.Grid.from_cfg(cfg); strat = Stratification(cfg, grid)
    setup = S2MFD.Setup(cfg, grid)
    sol = DynamicSolver(cfg, grid, strat, setup)
    sol.set_primitive_from_conserved(sol.conserved())
    return cfg, grid, strat, setup, sol

def max_lng(cs, S, ctx=None):
    """全半径・全波数での max ln|G| と、最悪半径を返す。"""
    cfg, grid, strat, setup, sol = ctx if ctx else build(cs)
    cfg.cfl_safety = S
    dt = sol.cfl_dt()
    m = grid.margin
    rr = grid.rr[m:grid.ixg-m]
    dl = np.minimum(grid.drr[m:grid.ixg-m], rr*grid.dth)
    cs_eff = strat.cs_eff[m:grid.ixg-m]
    nu_t = setup.nu_dif[m:grid.ixg-m]
    worst = -1e30; wr = None
    for k in range(len(rr)):
        nu = cs_eff[k]*dt/dl[k]                       # 移流 CFL 数
        d_sld = 0.5*(cs*cs_eff[k])*dl[k]              # SLD の最大拡散係数
        d = (nu_t[k] + d_sld*RESP)*dt/dl[k]**2        # 波数依存の拡散数
        z = -4.0*d*np.sin(0.5*TH)**2 - 1j*nu*np.sin(TH)
        g = np.log(np.abs(1.0 + z + 0.5*z*z)).max()
        if g > worst: worst, wr = g, rr[k]/cfg.RSUN
    return worst, wr, dt

def neutral_S(cs):
    ctx = build(cs)
    lo, hi = 0.005, 2.0
    if max_lng(cs, lo, ctx)[0] > 0: return np.nan, ctx
    for _ in range(40):
        mid = 0.5*(lo + hi)
        if max_lng(cs, mid, ctx)[0] <= 0: lo = mid
        else: hi = mid
    return lo, ctx

if __name__ == '__main__':
    print("SLD の実効拡散 (格子スケール = 1)")
    for L in (2, 3, 4, 6, 8, 12):
        i = np.argmin(abs(TH - 2*np.pi/L))
        print(f"  波長 {L:4.1f} セル: Phi_eff = {RESP[i]:.4f}")
    print(f"\n{'cs_factor':>10}{'中立 S0':>10}{'S=0.5 の ln|G|':>16}{'最悪半径':>10}{'dt(S=0.5)':>11}")
    for cs in (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50):
        S0, ctx = neutral_S(cs)
        g, wr, dt = max_lng(cs, 0.5, ctx)
        print(f"{cs:10.2f}{S0:10.3f}{g:16.2e}{wr:10.4f}{dt:11.1f}")
    print("\n2026-08-21 の実測: cs=0.30 で中立 S0=0.2、実際の崖は 0.8 (3.5 倍甘い)")
