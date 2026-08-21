"""wall 時間と CPU 時間の両方を測り、余分なスレッドの空回りを可視化する."""
import sys, os, time, numpy as np; sys.path.insert(0,'tests')
np.seterr(all='ignore')
import S2MFD, numba
from S2MFD.stratification import Stratification
from S2MFD.physics.dynamic import DynamicSolver
from conftest import make_cfg, make_grid
nx,ny = int(sys.argv[1]), int(sys.argv[2]); nstep = int(sys.argv[3])
cfg = make_cfg('parameters/rempel06.py', ix=nx, jx=ny, dynamics='hydro',
               angmom_bottom_bc='uniform_rotation')
grid=make_grid(cfg); strat=Stratification(cfg,grid); setup=S2MFD.Setup(cfg,grid)
sol=DynamicSolver(cfg,grid,strat,setup)
sol.set_primitive_from_conserved(sol.conserved()); dt=sol.cfl_dt()
for _ in range(30): sol.step(dt)
w0=time.time(); c0=time.process_time()
for _ in range(nstep): sol.step(dt)
w=time.time()-w0; c=time.process_time()-c0
print(f"{nx}x{ny} nb={numba.get_num_threads()} OMP={os.environ.get('OMP_NUM_THREADS','-')} "
      f"wall={w/nstep*1e3:6.2f} ms/step  cpu={c/nstep*1e3:7.2f} ms/step  "
      f"占有コア={c/w:5.2f}")
