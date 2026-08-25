"""磁場の SLD フィルタが散逸させるエネルギーを直接測る。

Rempel (2006) 式 (22): dE_B/dt = Q_L^Omega + Q_L^M - Q_eta
論文はこれが閉じるが、本実装は磁場にも人工拡散 (SLD フィルタ) を掛けて
いるので、その分だけ余分な散逸項が要る:

    dE_B/dt = Q_L^Omega + Q_L^M - Q_eta - Q_SLD

E_B = int dV B_phi^2/(8 pi) なので、フィルタによる変化率は

    Q_SLD = - int dV (B_phi/4pi) (dB_phi/dt)_filter

フィルタは 1 ステップ分の増分 dt*d を足すので、(dB_phi/dt) は
(filter 後 - filter 前)/dt で取れる。
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
np.seterr(all='ignore')
import S2MFD
from S2MFD.stratification import Stratification
from S2MFD.physics.energy import EnergyBudget
from S2MFD.physics import poloidal_from_potential
from S2MFD.physics.dynamic import DynamicSolver

tag, alpha0 = sys.argv[1], float(sys.argv[2])
cfg = S2MFD.build_cfg('parameters/rempel06_paper.py', alpha0=alpha0, magnetic_buoyancy=1,
               sld_cs_factor=0.30, alpha_quenching=False)
grid = S2MFD.Grid.from_cfg(cfg); strat = Stratification(cfg, grid); setup = S2MFD.Setup(cfg, grid)
sol = DynamicSolver(cfg, grid, strat, setup); eb = EnergyBudget(cfg, grid, strat, setup)
m = grid.margin
d = np.load(f'results_rempel/{tag}/final_state.npz')
for k in ('om1', 'vrr', 'vth', 'ro1', 'se1'):
    getattr(sol, k)[:] = d[k]
Bph = np.ascontiguousarray(d['Bph']); Aph = np.ascontiguousarray(d['Aph'])
sol.magnetic = True
sol.set_primitive_from_conserved(sol.conserved()); sol.sync_to_induction()
pm = poloidal_from_potential(Aph, grid)
sol.set_magnetic_field(pm[0], pm[1], Bph)
dt = sol.cfl_dt()

q = eb.exchanges(sol.om1, sol.vrr, sol.vth, pm[0], pm[1], Bph, sol.se1)
ql = q['Q_Lambda']

# フィルタを 1 回だけ掛けて増分を測る
B0 = Bph.copy(); A0 = Aph.copy()
B1, A1 = sol.magnetic_filter(Bph.copy(), Aph.copy(), dt)
dB = (B1 - B0)/dt
# トロイダル磁場のエネルギー変化率 (符号: 散逸なら正)
q_sld = -eb._integrate(B0*dB/(4.0*np.pi))

# ポロイダル側も測る (E_pol = int (Br^2+Bth^2)/(8pi))
pm0 = poloidal_from_potential(A0, grid)
pm1 = poloidal_from_potential(A1, grid)
q_sld_pol = -eb._integrate((pm0[0]*(pm1[0]-pm0[0]) + pm0[1]*(pm1[1]-pm0[1]))
                           / dt/(4.0*np.pi))

print(f"=== {tag} (alpha0={alpha0/100:.3f} m/s), dt={dt:.1f}s ===")
print(f"Q_Lambda = {ql:.4e} erg/s\n")
print(f"{'項':28}{'erg/s':>13}{'Q_Lambda 比':>13}")
for k, v in (('Q_L^Omega (差動回転 -> 磁場)', q['Q_L_Omega']),
             ('Q_L^M (子午面流 -> 磁場)', q['Q_L_M']),
             ('Q_eta (オーム散逸)', q['Q_eta']),
             ('Q_SLD トロイダル (実測)', q_sld),
             ('Q_SLD ポロイダル (参考)', q_sld_pol)):
    print(f"{k:28}{v:13.4e}{v/ql:13.4f}")
gap = q['Q_L_Omega'] + q['Q_L_M'] - q['Q_eta']
print(f"\n収支の未計上分 (入力 - Q_eta) = {gap:.4e} = {gap/ql:.4f} Q_Lambda")
print(f"フィルタの実測散逸           = {q_sld:.4e} = {q_sld/ql:.4f} Q_Lambda")
print(f"-> 未計上分の {q_sld/gap*100:.0f}% を説明する")
print(f"\n残差 = {(gap-q_sld)/ql:+.4f} Q_Lambda "
      f"(入力の {abs(gap-q_sld)/(q['Q_L_Omega']+q['Q_L_M'])*100:.0f}%)")
