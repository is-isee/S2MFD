"""S2MFD_PARALLEL の設定違いで結果がビット一致するかを見る."""
import sys, os, hashlib, numpy as np; sys.path.insert(0,'tests')
np.seterr(all='ignore')
import S2MFD
from S2MFD.stratification import Stratification
from S2MFD.physics.dynamic import DynamicSolver
from S2MFD.physics._jit import PARALLEL
from conftest import make_cfg, make_grid
nx,ny,nstep = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
cfg = make_cfg('parameters/rempel06.py', ix=nx, jx=ny, dynamics='hydro',
               angmom_bottom_bc='uniform_rotation')
grid=make_grid(cfg); strat=Stratification(cfg,grid); setup=S2MFD.Setup(cfg,grid)
sol=DynamicSolver(cfg,grid,strat,setup)
sol.set_primitive_from_conserved(sol.conserved()); dt=sol.cfl_dt()
for _ in range(nstep): sol.step(dt)
h = hashlib.sha256()
for k in ('om1','vrr','vth','ro1','se1'):
    h.update(np.ascontiguousarray(getattr(sol,k)).tobytes())
print(f"S2MFD_PARALLEL={os.environ.get('S2MFD_PARALLEL','(未設定)'):>9} "
      f"PARALLEL={PARALLEL} sha256={h.hexdigest()[:32]}")
