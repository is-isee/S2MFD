"""Rempel 2006 参照モデルの流体緩和 (Lambda 効果による差動回転の形成)."""
import sys, os, time, numpy as np; sys.path.insert(0,'tests')
np.seterr(all='ignore')
import S2MFD
PARFILE=os.environ.get('S2MFD_PARFILE','parameters/rempel06.py')
from S2MFD.stratification import Stratification
from S2MFD.physics import conservative as cons
from S2MFD.physics.dynamic import DynamicSolver
from S2MFD.physics.energy import EnergyBudget, solar_luminosity
from conftest import make_cfg, make_grid

nx,ny = int(sys.argv[1]), int(sys.argv[2])
years = float(sys.argv[3]); bc = sys.argv[4]; tag = sys.argv[5]
outdir = f'results_rempel/{tag}'; os.makedirs(outdir, exist_ok=True)

over = {}; init = None
for a in sys.argv[6:]:
    k, v = a.split('=')
    if k == 'init': init = v
    else:
        # margin のように int でなければならない値がある
        try: over[k] = int(v)
        except ValueError: over[k] = float(v)
cfg = make_cfg(PARFILE, ix=nx, jx=ny, dynamics='hydro',
               angmom_bottom_bc=bc, **over)
grid=make_grid(cfg); strat=Stratification(cfg,grid); setup=S2MFD.Setup(cfg,grid)
sol = DynamicSolver(cfg,grid,strat,setup); eb = EnergyBudget(cfg,grid,strat,setup)
m=grid.margin; sl=(slice(m,grid.ixg-m),slice(m,grid.jxg-m))
if init:                      # 別解像度の緩和済み状態から始める
    dd = np.load(init)
    for k in ('om1','vrr','vth','ro1','se1'):
        getattr(sol, k)[:] = dd[k]
    print(f"[{tag}] {init} から開始", flush=True)
sol.set_primitive_from_conserved(sol.conserved())
dt = sol.cfl_dt(); nsteps=int(years*3.156e7/dt)
print(f"[{tag}] overrides={over}", flush=True)
print(f"[{tag}] {nx}x{ny} BC={bc} rmax={grid.rrmax/cfg.RSUN:.3f}R dt={dt:.1f}s "
      f"{years}yr = {nsteps} steps", flush=True)

zeta2=strat.zeta**2
L0=cons.cell_integral(strat.JL*sol.om1,grid.drr,grid.dth,m)
M0=cons.cell_integral(strat.JM*sol.ro1*zeta2[:,None],grid.drr,grid.dth,m)
scaleL=cons.cell_integral(np.abs(strat.JL*cfg.om0),grid.drr,grid.dth,m)
scaleM=cons.cell_integral(np.abs(strat.JM*strat.ro0[:,None]),grid.drr,grid.dth,m)
th=grid.th[m:grid.jxg-m]; eq=np.argmin(abs(th-np.pi/2))
hist=[]; t0=time.time(); t=0.0; nout=max(1,nsteps//500)
for n in range(1, nsteps+1):
    sol.step(dt); t += dt
    if n % 2000 == 0:      # 流れが育つので dt を追随させる
        dt = min(dt, sol.cfl_dt())
    if n % nout == 0:
        om=sol.om1[sl]
        L=cons.cell_integral(strat.JL*sol.om1,grid.drr,grid.dth,m)
        M=cons.cell_integral(strat.JM*sol.ro1*zeta2[:,None],grid.drr,grid.dth,m)
        q=eb.exchanges(sol.om1,sol.vrr,sol.vth,sol.brr,sol.bth,sol.bph,sol.se1)
        e=eb.reservoirs(sol.om1,sol.vrr,sol.vth,sol.bph)
        dr=(om[-1,eq]-om[-1,0])/cfg.om0
        hist.append([t,dr,np.abs(om).max()/cfg.om0,
                     np.abs(sol.vrr[sl]).max()/100,np.abs(sol.vth[sl]).max()/100,
                     np.abs(sol.se1[sl]).max(),
                     (L-L0-sol.boundary_angmom_flux)/scaleL,(M-M0)/scaleM,
                     e[0],e[1],eb.differential_rotation_energy(sol.om1),
                     q['Q_Lambda'],q['Q_nu_Omega'],q['Q_C'],q['Q_nu_M'],q['Q_B']])
        if not np.all(np.isfinite(sol.om1)):
            print(f"[{tag}] *** 発散 step {n} ***", flush=True); break
        if n % (nout*25) == 0:
            print(f"[{tag}] {n:>9}/{nsteps} t={t/3.156e7:7.3f}yr DR={dr:+.4f} "
                  f"vr={hist[-1][3]:5.2f} vth={hist[-1][4]:6.2f}m/s "
                  f"QL={q['Q_Lambda']/solar_luminosity:.4f}Lsun "
                  f"収支残差={hist[-1][6]:+.2e} ({(time.time()-t0)/n*1e3:.2f}ms/st)",
                  flush=True)
        np.savez(f'{outdir}/history.npz', hist=np.array(hist))
        np.savez(f'{outdir}/state.npz', om1=sol.om1, vrr=sol.vrr, vth=sol.vth,
                 ro1=sol.ro1, se1=sol.se1, rr=grid.rr, th=grid.th, t=t,
                 om0=cfg.om0, RSUN=cfg.RSUN, margin=m, drr=grid.drr, dth=grid.dth)
print(f"[{tag}] done {(time.time()-t0)/60:.1f} min", flush=True)
om=sol.om1[sl]
print(f"[{tag}] 表面差動回転 (Om_eq-Om_pole)/Om0 = {(om[-1,eq]-om[-1,0])/cfg.om0:+.4f} "
      f"(Rempel 表1 参照モデル: 0.27)", flush=True)
print(eb.report(sol.om1,sol.vrr,sol.vth,sol.brr,sol.bth,sol.bph,sol.se1), flush=True)
