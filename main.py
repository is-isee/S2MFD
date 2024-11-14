import matplotlib.pyplot as plt
import numpy as np
import os, glob
from FLD_ISEE.tools import drr1, drr2, dth1, dth2
import FLD_ISEE.config as cfg
import importlib
from FLD_ISEE import grid_c, setup_c
from numba import njit

# configを強制的に再読み込み
importlib.reload(cfg)

if not os.path.isfile(cfg.datadir+'data.000001.npz'):
   cfg.cont_flag = False

os.makedirs(cfg.datadir,exist_ok=True)
      
if cfg.cont_flag:
   pass
   grid = grid_c.load(cfg.gridfile)
   setup = setup_c.load(cfg.setupfile)
else:
   grid = grid_c(ix=cfg.ix,jx=cfg.jx,margin=cfg.margin
              ,rrmin=cfg.rrmin,rrmax=cfg.rrmax,thmin=cfg.thmin,thmax=cfg.thmax)
   grid.save(cfg.gridfile)
   setup = setup_c(cfg,grid)
   setup.save(cfg.setupfile)

def time_marching(Bph, Aph, dt, grid, setup):

   # poloidal magnetic field
   Brr = + dth2(grid.sinTH*Aph,grid.dth)/grid.RR/grid.sinTH
   Bth = - drr2(   grid.RR*Aph,grid.drr)/grid.RR
      
   Bph_adrr = - drr2(grid.RR*setup.urr*Bph,grid.drr)/grid.RR #  動径方向の移流(Bph)
   Bph_adth = - dth2(        setup.uth*Bph,grid.dth)/grid.RR #  緯度方向の移流(Bph)
   
   Aph_adrr = - drr2(   grid.RR*Aph,grid.drr)*setup.urr/grid.RR            # 動径方向の移流(Aph)
   Aph_adth = - dth2(grid.sinTH*Aph,grid.dth)*setup.uth/grid.RR/grid.sinTH # 緯度方向の移流(Aph)
   
   # 磁場の微分(拡散量)
   Bphrr = drr1(Bph,grid.drr,'up')
   Bphth = dth1(Bph,grid.dth,'up')
   Aphrr = drr1(Aph,grid.drr,'up')
   Aphth = dth1(Aph,grid.dth,'up')
   
   Bph_dfrr = setup.et*drr1(grid.RRm**2*Bphrr,grid.drr,'dw')/grid.RR**2
   Bph_dfth = setup.et*dth1(grid.sinTHm*Bphth,grid.dth,'dw')/grid.RR**2/grid.sinTH
   
   Aph_dfrr = setup.et*drr1(grid.RRm**2*Aphrr,grid.drr,'dw')/grid.RR**2
   Aph_dfth = setup.et*dth1(grid.sinTHm*Aphth,grid.dth,'dw')/grid.RR**2/grid.sinTH
   
   # diffusivity gradient influence
   Bph_dfrrg = setup.etrr*drr1(grid.RR*Bph,grid.drr,'dw')/grid.RR
   
   # Omega effect
   Bph_omrr = Brr*setup.omrr*grid.RR*grid.sinTH
   Bph_omth = Bth*setup.omth*grid.RR*grid.sinTH
   
   # source term
   tmp, Bphso = np.meshgrid(grid.rr,Bph[setup.ibase,:], indexing='ij')
   Aph_sour = setup.so*Bphso/(1 + (Bphso)**2)

   dBph = + (Bph_adrr + Bph_adth) \
          + (Bph_dfrr + Bph_dfth + Bph_dfrrg - setup.et*Bph/grid.RR**2/grid.sinTH**2) \
          + (Bph_omrr + Bph_omth) 
          
   dAph = + (Aph_adrr + Aph_adth) \
          + (Aph_dfrr + Aph_dfth - setup.et*Aph/grid.RR**2/grid.sinTH**2) \
          + Aph_sour
      
   Bphm = Bph + dt*dBph
   Aphm = Aph + dt*dAph
   
   return Bphm, Aphm

#
def boundary_condition(Aph, Bph, grid):
   # 動径方向境界条件
   for i in range(0, grid.margin):
      #下部境界条件(完全導体)A=B=0
      Aph[i,grid.margin:grid.jxg-grid.margin] = - Aph[2*grid.margin-i-1,grid.margin:grid.jxg-grid.margin]
      Bph[i,grid.margin:grid.jxg-grid.margin] = + Bph[2*grid.margin-i-1,grid.margin:grid.jxg-grid.margin] \
         /grid.rr[i]*grid.rr[2*grid.margin-i-1]

      #上部境界条件B=0,dA/dr=0
      Bph[grid.ixg-i-1,grid.margin:grid.jxg-grid.margin] = -Bph[grid.ixg-2*grid.margin+i,grid.margin:grid.jxg-grid.margin]
      Aph[grid.ixg-i-1,grid.margin:grid.jxg-grid.margin] = +Aph[grid.ixg-2*grid.margin+i,grid.margin:grid.jxg-grid.margin] \
         /grid.rr[grid.ixg-i-1]*grid.rr[grid.ixg-2*grid.margin+i]

   # 緯度方向境界条件      
   for j in range(0, grid.margin):
      #極の境界条件A=B=0
      Bph[grid.margin:grid.ixg-grid.margin,j]  = -Bph[grid.margin:grid.ixg-grid.margin,2*grid.margin-j-1]
      Aph[grid.margin:grid.ixg-grid.margin,j]  = -Aph[grid.margin:grid.ixg-grid.margin,2*grid.margin-j-1]
      #赤道境界条件B=0(反対称),dA/dθ=0
      Bph[grid.margin:grid.ixg-grid.margin,grid.jxg-j-1] = -Bph[grid.margin:grid.ixg-grid.margin,grid.jxg-2*grid.margin+j]
      Aph[grid.margin:grid.ixg-grid.margin,grid.jxg-j-1] = -Aph[grid.margin:grid.ixg-grid.margin,grid.jxg-2*grid.margin+j]
   return Aph, Bph
# Prepare data directory
os.makedirs('data',exist_ok=True)

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
   #Aph = np.zeros((grid.ixg, grid.jxg))
   #Aph = sinTH/(RR/rsun)**2
   Bph = np.sin(2*grid.TH)*0.1
   Bph[0:setup.ibase,:] = 0

   time = 0

Brr = + dth2(grid.sinTH*Aph,grid.dth)/grid.RR/grid.sinTH
Bth = - drr2(   grid.RR*Aph,grid.drr)/grid.RR

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
      ax.set_xlim(0,1)
      ax.set_ylim(-1,1)
      plt.pause(0.01)
      print(time/86400,n,nd)
      Brr =  dth2(grid.sinTH*Aph,grid.dth)/grid.RR/grid.sinTH
      Bth = -drr2(   grid.RR*Aph,grid.drr)/grid.RR
      np.savez(file='data/data.'+str(nd).zfill(6)+'.npz' \
            ,Aph=Aph,Bph=Bph,Brr=Brr,Bth=Bth,time=time)
                
   ####ダイナモ方程式                       
   Bphm, Aphm = time_marching(Bph , Aph ,dt, grid, setup)
   
   Aphm, Bphm = boundary_condition(Aphm, Bphm, grid)

   Bphn, Aphn = time_marching(Bphm, Aphm, dt, grid, setup)
   Aphn, Bphn = boundary_condition(Aphn, Bphn, grid)
    
   Bph = 0.5*Bph + 0.5*Bphn
   Aph = 0.5*Aph + 0.5*Aphn

   