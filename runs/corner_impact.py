"""4 隅ゴーストの修正が結果にどれだけ効くかを測る。

修正前の挙動は「境界条件が角を一度も書かない」= **角が初期値のまま凍結**
なので、修正版のコードで毎ステップ角を初期値に戻せば正確に再現できる。
同じ初期状態から両方を走らせて発散の仕方を見る。

使い方: python runs/corner_impact.py <init.npz> <alpha0> <years>
"""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
np.seterr(all='ignore')
import S2MFD
from S2MFD.stratification import Stratification
from S2MFD.physics import time_marching, poloidal_mag, boundary_condition
from S2MFD.physics.dynamic import DynamicSolver

init, alpha0, years = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])

def run(freeze_corners):
    cfg = S2MFD.build_cfg('parameters/rempel06_paper.py', alpha0=alpha0,
                   magnetic_buoyancy=1, sld_cs_factor=0.30, alpha_quenching=False)
    grid = S2MFD.Grid.from_cfg(cfg); strat = Stratification(cfg, grid)
    setup = S2MFD.Setup(cfg, grid)
    sol = DynamicSolver(cfg, grid, strat, setup)
    m = grid.margin
    d = np.load(init)
    for k in ('om1', 'vrr', 'vth', 'ro1', 'se1'):
        getattr(sol, k)[:] = d[k]
    Bph = np.ascontiguousarray(d['Bph']); Aph = np.ascontiguousarray(d['Aph'])
    sol.magnetic = True
    sol.set_primitive_from_conserved(sol.conserved()); sol.sync_to_induction()
    # 修正前を再現するために、初期状態の角を控えておく
    cs = [(slice(None, m), slice(None, m)), (slice(None, m), slice(-m, None)),
          (slice(-m, None), slice(None, m)), (slice(-m, None), slice(-m, None))]
    # time_marching は新しい配列を返すので id() を鍵にしてはいけない
    frozen_B = [Bph[s].copy() for s in cs]
    frozen_A = [Aph[s].copy() for s in cs]
    pm = poloidal_from_potential(Aph, grid)
    sol.set_magnetic_field(pm[0], pm[1], Bph)
    dt = sol.cfl_dt()
    n = int(years*3.156e7/dt)
    sl = (slice(m, grid.ixg-m), slice(m, grid.jxg-m))
    out = []
    t0 = time.time()
    for k in range(1, n+1):
        Bph, Aph = time_marching(Bph, Aph, dt, cfg, grid, setup)
        Bph, Aph = sol.magnetic_filter(Bph, Aph, dt)
        Bph, Aph = boundary_condition(Bph, Aph, cfg, grid, None)
        if freeze_corners:                    # 修正前の挙動を再現
            for a, fr in ((Bph, frozen_B), (Aph, frozen_A)):
                for s, v in zip(cs, fr):
                    a[s] = v
        pm = poloidal_from_potential(Aph, grid)
        sol.set_magnetic_field(pm[0], pm[1], Bph)
        sol.step(dt); sol.sync_to_induction()
        if k % max(1, n//40) == 0:
            om = sol.om1[sl]
            out.append((k*dt/3.156e7, np.abs(Bph[sl]).max()*1e-4,
                        float(om[-1, -1] - om[-1, 0])/cfg.om0,
                        sol.vth[sl].copy(), Bph[sl].copy()))
    print(f"  {'凍結(修正前)' if freeze_corners else '修正版'}: "
          f"{n} step, {(time.time()-t0)/60:.1f} 分", flush=True)
    return out, dt

new, dt = run(False)
old, _ = run(True)
print(f"\ndt={dt:.1f}s  {years} 年 = {int(years*3.156e7/dt)} step")
print(f"\n{'t[yr]':>7}{'|Bph|max 修正版':>16}{'凍結':>12}{'相対差':>10}"
      f"{'DR 修正版':>12}{'凍結':>10}{'相対差':>10}{'vth 場の相対差':>16}")
for (t1, b1, d1, v1, f1), (t2, b2, d2, v2, f2) in zip(new, old):
    rv = np.abs(v1-v2).max()/max(np.abs(v1).max(), 1e-300)
    print(f"{t1:7.2f}{b1:16.5f}{b2:12.5f}{abs(b1-b2)/b1:10.2e}"
          f"{d1:12.5f}{d2:10.5f}{abs(d1-d2)/abs(d1):10.2e}{rv:16.2e}")
