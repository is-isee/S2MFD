"""108x144 で 1 ステップあたりの実時間を測る (スレッド数依存の確認)."""
import sys, os, time, numpy as np; sys.path.insert(0,'tests')
np.seterr(all='ignore')
import S2MFD
from S2MFD.stratification import Stratification
from S2MFD.physics.dynamic import DynamicSolver
from conftest import make_cfg, make_grid
import numba
nx,ny = int(sys.argv[1]), int(sys.argv[2]); nstep = int(sys.argv[3])
cfg = make_cfg('parameters/rempel06.py', ix=nx, jx=ny, dynamics='hydro',
               angmom_bottom_bc='uniform_rotation')
grid=make_grid(cfg); strat=Stratification(cfg,grid); setup=S2MFD.Setup(cfg,grid)
sol=DynamicSolver(cfg,grid,strat,setup)
sol.set_primitive_from_conserved(sol.conserved()); dt=sol.cfl_dt()
for _ in range(20): sol.step(dt)          # JIT ウォームアップ
t0=time.time()
for _ in range(nstep): sol.step(dt)
el=time.time()-t0
print(f"{nx}x{ny} threads={numba.get_num_threads()} {el/nstep*1e3:.2f} ms/step")
