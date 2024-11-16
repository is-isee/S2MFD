import matplotlib.pyplot as plt
import numpy as np
import os, glob, importlib
import S2MFD.config as cfg
import S2MFD

# configを強制的に再読み込み
importlib.reload(cfg)

if not os.path.isfile(cfg.datadir+'data.000000.npz'):
   cfg.cont_flag = False

os.makedirs(cfg.datadir,exist_ok=True)
      
if cfg.cont_flag:
   grid = S2MFD.grid_c.load(cfg.gridfile)
   setup = S2MFD.setup_c.load(cfg.setupfile)
else:
   grid = S2MFD.grid_c(ix=cfg.ix,jx=cfg.jx,margin=cfg.margin
              ,rrmin=cfg.rrmin,rrmax=cfg.rrmax,thmin=cfg.thmin,thmax=cfg.thmax)
   grid.save(cfg.gridfile)
   setup = S2MFD.setup_c(cfg,grid)
   setup.save(cfg.setupfile)

# Prepare data directory
os.makedirs(cfg.datadir,exist_ok=True)

#CFL condition
c_cfl=0.1
dtmin = 1.e10
for i in range(grid.margin,grid.ixg-grid.margin):
   for j in range(grid.margin,grid.jxg-grid.margin):
      dt_adv = c_cfl * np.min([grid.drr, grid.rr[i]*grid.dth])/ np.sqrt(setup.urr[i,j]**2 + setup.uth[i,j]**2)
      dt_dif = c_cfl * np.min([grid.drr, grid.rr[i]*grid.dth])**2/(2 * setup.et[i,j])
      dtmin = np.min([dtmin,dt_adv,dt_dif])
      
dt  = dtmin

# 初期条件
if cfg.cont_flag:
   files = glob.glob('data/data.*.npz')
   nd = max([int(f.split('.')[-2]) for f in files])
   d = np.load(file='data/data.'+str(nd).zfill(6)+'.npz')
   Aph = d['Aph']
   Bph = d['Bph']
   time = d['time']
else:
   nd = 0
   Aph = np.zeros((grid.ixg, grid.jxg))
   Bph = np.zeros((grid.ixg, grid.jxg))
   Aph = grid.sinTH/(grid.RR/cfg.RSUN)**2
   Aph[0:setup.ibase,:] = 0
   #Bph = np.sin(2*grid.TH)*0.1
   #Bph[0:setup.ibase,:] = 0

   time = 0

Brr, Bth = S2MFD.poloidal_mag(Aph, grid.RR, grid.sinTH, grid.drr, grid.dth)
np.savez(file='data/data.'+str(nd).zfill(6)+'.npz' \
            ,Aph=Aph,Bph=Bph,Brr=Brr,Bth=Bth,time=time)

n = 0

plt.clf()
plt.close('all')
fig = plt.figure('dynamo',figsize=(5,10))

while time < cfg.tend:
   time += dt
   n += 1
   if(time//cfg.dtout != (time-dt)//cfg.dtout):
      nd += 1
      ax = fig.add_subplot(111,aspect='equal')
      ax.pcolormesh(grid.Y/cfg.RSUN,grid.X/cfg.RSUN,Bph,vmax=1.e0,vmin=-1.e0,cmap='bwr',shading='auto')
      ax.contour(grid.Y/cfg.RSUN,grid.X/cfg.RSUN,grid.RR/cfg.RSUN*grid.sinTH*Aph,colors='black',levels=np.linspace(-8.e12,8.e12,10))
      ax.set_xlim( 0,1)
      ax.set_ylim(-1,1)
      plt.pause(0.01)
      #print(time/86400,n,nd)
      print(f"{time/86400:7.1f} [day]; n={n:06d}; nd={nd:04d}")
      Brr, Bth = S2MFD.poloidal_mag(Aph, grid.RR, grid.sinTH, grid.drr, grid.dth)
      np.savez(file='data/data.'+str(nd).zfill(6)+'.npz' \
            ,Aph=Aph,Bph=Bph,Brr=Brr,Bth=Bth,time=time)
                
   #### dynamo equation                  
   Bphm, Aphm = S2MFD.time_marching(Bph , Aph ,dt, grid, setup)
   Aphm, Bphm = S2MFD.boundary_condition(Aphm, Bphm, grid)

   Bphn, Aphn = S2MFD.time_marching(Bphm, Aphm, dt, grid, setup)
   Aphn, Bphn = S2MFD.boundary_condition(Aphn, Bphn, grid)
    
   Bph = 0.5*Bph + 0.5*Bphn
   Aph = 0.5*Aph + 0.5*Aphn

   