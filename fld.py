import matplotlib.pyplot as plt
import numpy as np
import pickle
import os, sys, glob
from scipy.special import erf
from tools import drr1, drr2, dth1, dth2
from dataclasses import dataclass, field
import config as cfg
import importlib

# configを強制的に再読み込み
importlib.reload(cfg)

if not os.path.isfile(cfg.datadir+'data.000001.npz'):
   cfg.cont_flag = False

os.makedirs(cfg.datadir,exist_ok=True)

@dataclass
class grid_c:
   ix: int
   jx: int
   ixg: int = field(init=False)
   jxg: int = field(init=False)
   margin: int
   rrmin: float
   rrmax: float
   thmin: float
   thmax: float
   drr: float = field(init=False)
   dth: float = field(init=False)
   rr: np.ndarray = field(init=False)
   th: np.ndarray = field(init=False)
   
   RR: np.ndarray = field(init=False)
   TH: np.ndarray = field(init=False)
   RRm: np.ndarray = field(init=False) 
   THm: np.ndarray = field(init=False)
   sinTH: np.ndarray = field(init=False)
   cosTH: np.ndarray = field(init=False)   
   X: np.ndarray = field(init=False)
   Y: np.ndarray = field(init=False)
   
   def __post_init__(self):
      # dr,dθの設定
      self.ixg = self.ix + 2*self.margin
      self.jxg = self.jx + 2*self.margin
      self.drr = (self.rrmax - self.rrmin)/self.ix
      self.dth = (self.thmax - self.thmin)/self.jx

      #座標rrの設定
      self.rr = np.zeros(self.ixg)
      self.rr[0] = self.rrmin + self.drr*(0.5 - self.margin)

      for i in range(1, self.ixg):
         self.rr[i] = self.rr[i - 1] + self.drr
   
      #座標thの設定    
      self.th    = np.zeros(self.jxg)
      self.th[0] = self.thmin + self.dth*(0.5 - self.margin)

      for j in range(1,self.jxg):
         self.th[j] = self.th[j - 1] + self.dth
         
      self.RR ,self.TH  = np.meshgrid(self.rr, self.th,indexing='ij')
      self.RRm = np.zeros_like(self.RR)
      self.THm = np.zeros_like(self.TH)
      
      self.RRm[1:self.ixg,:] = 0.5*(self.RR[1:self.ixg,:] + self.RR[0:self.ixg-1,:])
      self.THm[:,1:self.jxg] = 0.5*(self.TH[:,1:self.jxg] + self.TH[:,0:self.jxg-1])
      
      self.sinTH = np.sin(self.TH)
      self.cosTH = np.cos(self.TH)
      self.sinTHm = np.sin(self.THm)
      
      self.X, self.Y = self.RR * np.cos(self.TH), self.RR * np.sin(self.TH)
   
   def save(self, filename):
      with open(filename, 'wb') as f:
         pickle.dump(self, f)
      print('grid-c instance saved to', filename)
   
   @classmethod
   def load(cls, filename):
      with open(filename, 'rb') as f:
         return pickle.load(f)
      
if cfg.cont_flag:
   grid = grid_c.load(cfg.gridfile)
else:
   grid = grid_c(ix=cfg.ix,jx=cfg.jx,margin=cfg.margin
              ,rrmin=cfg.rrmin,rrmax=cfg.rrmax,thmin=cfg.thmin,thmax=cfg.thmax)
   grid.save(cfg.gridfile)

def time_marching(Bph, Aph, dt,urr, uth,grid,et,etrr,so,omrr,omth,ibase):

   # poloidal magnetic field
   Brr = + dth2(grid.sinTH*Aph,grid.dth)/grid.RR/grid.sinTH
   Bth = - drr2(   grid.RR*Aph,grid.drr)/grid.RR
      
   Bph_adrr = - drr2(grid.RR*urr*Bph,grid.drr)/grid.RR #  動径方向の移流(Bph)
   Bph_adth = - dth2(        uth*Bph,grid.dth)/grid.RR #  緯度方向の移流(Bph)
   
   Aph_adrr = - drr2(   grid.RR*Aph,grid.drr)*urr/grid.RR            # 動径方向の移流(Aph)
   Aph_adth = - dth2(grid.sinTH*Aph,grid.dth)*uth/grid.RR/grid.sinTH # 緯度方向の移流(Aph)
   
   # 磁場の微分(拡散量)
   Bphrr = drr1(Bph,grid.drr,'up')
   Bphth = dth1(Bph,grid.dth,'up')
   Aphrr = drr1(Aph,grid.drr,'up')
   Aphth = dth1(Aph,grid.dth,'up')
   
   Bph_dfrr = et*drr1(grid.RRm**2*Bphrr,grid.drr,'dw')/grid.RR**2
   Bph_dfth = et*dth1(grid.sinTHm*Bphth,grid.dth,'dw')/grid.RR**2/grid.sinTH
   Bph_dfrrg = etrr*drr1(grid.RR*Bph,grid.drr,'dw')/grid.RR
   Aph_dfrr = et*drr1(grid.RRm**2*Aphrr,grid.drr,'dw')/grid.RR**2
   Aph_dfth = et*dth1(grid.sinTHm*Aphth,grid.dth,'dw')/grid.RR**2/grid.sinTH
   
   # Omega effect
   Bph_omrr = Brr*omrr*grid.RR*grid.sinTH
   Bph_omth = Bth*omth*grid.RR*grid.sinTH
   
   # source term
   tmp, Bphso = np.meshgrid(grid.rr,Bph[ibase,:], indexing='ij')
   Aph_sour = so*Bphso/(1 + (Bphso)**2)

   dBph = + (Bph_adrr + Bph_adth) \
          + (Bph_dfrr + Bph_dfth + Bph_dfrrg - et*Bph/grid.RR**2/grid.sinTH**2) \
          + (Bph_omrr + Bph_omth) 
          
   dAph = + (Aph_adrr + Aph_adth) \
          + (Aph_dfrr + Aph_dfth - et*Aph/grid.RR**2/grid.sinTH**2) \
          + Aph_sour
      
   Bphm = Bph + dt*dBph
   Aphm = Aph + dt*dAph
   
   return Bphm, Aphm

#
def boundary_condition(Aph, Bph,margin,rr,th,ixg,jxg):
   # 動径方向境界条件
   for i in range(0, margin):
      #下部境界条件(完全導体)A=B=0
      Bph[i,margin:jxg-margin] = + Bph[2*margin-i-1,margin:jxg-margin]/rr[i]*rr[2*margin-i-1]
      Aph[i,margin:jxg-margin] = - Aph[2*margin-i-1,margin:jxg-margin]

      #上部境界条件B=0,dA/dr=0
      Bph[ixg-i-1,margin:jxg-margin] = -Bph[ixg-2*margin+i,margin:jxg-margin]
      Aph[ixg-i-1,margin:jxg-margin] = +Aph[ixg-2*margin+i,margin:jxg-margin]/rr[ixg-i-1]*rr[ixg-2*margin+i]

   # 緯度方向境界条件      
   for j in range(0, margin):
      #極の境界条件A=B=0
      Bph[margin:ixg-margin,j]  = -Bph[margin:ixg-margin,2*margin-j-1]
      Aph[margin:ixg-margin,j]  = -Aph[margin:ixg-margin,2*margin-j-1]
      #赤道境界条件B=0(反対称),dA/dθ=0
      Bph[margin:ixg-margin,jxg-j-1] = -Bph[margin:ixg-margin,jxg-2*margin+j]
      Aph[margin:ixg-margin,jxg-j-1] = -Aph[margin:ixg-margin,jxg-2*margin+j]
   return Aph, Bph
# Prepare data directory
os.makedirs('data',exist_ok=True)

##########################
##########################
# differential rotation
ome = 456.e-9*2*np.pi # rotation rate at equator
omc = 0.92*ome         # rotation rate at radiative zone
rrc = 0.7*cfg.RSUN          # radiative zone boundary
d  = 0.02*cfg.RSUN     # width of tachocline
c2 = 0.2*ome

om = omc + 0.5*(1 + erf((grid.RR-rrc)/d))*(ome - omc - c2*grid.cosTH**2)

omrr = drr2(om, grid.drr)
omth = dth2(om, grid.dth)/grid.RR

# diffusivity
etc = 1.e9
ett = 1.e11

et = etc + 0.5*(ett - etc)*(1 + erf((grid.RR-rrc)/d))

etrr = drr2(et, grid.drr)

#タコクラインの要素番号
ibase = np.argmin(abs(grid.rr - rrc))

cso = 35
so0 = cso*ett/cfg.RSUN
r1 = 0.95*cfg.RSUN
d1 = 0.01*cfg.RSUN

so = so0*0.5*(1 + erf((grid.RR-r1)/d1))*(1 - erf(grid.RR-cfg.RSUN)/d1)*grid.cosTH*grid.sinTH

# Meridional flow
u0 = 1000
rb = 0.65*cfg.RSUN

urr = -u0*2*(cfg.RSUN - rb)/np.pi/grid.RR \
     *(grid.RR-rb)**2/(cfg.RSUN - rb)**2 \
     *np.sin(np.pi*(grid.RR-rb)/(cfg.RSUN-rb))*(3*grid.cosTH**2 - 1)
     
uth = u0*((3*grid.RR-rb)/(cfg.RSUN-rb)*np.sin(np.pi*(grid.RR-rb)/(cfg.RSUN-rb)) \
      + grid.RR*np.pi/(cfg.RSUN-rb)*(grid.RR-rb)/(cfg.RSUN-rb)*np.cos(np.pi*(grid.RR-rb)/(cfg.RSUN-rb))) \
      *2*(cfg.RSUN-rb)/np.pi/grid.RR*(grid.RR-rb)/(cfg.RSUN-rb)*grid.cosTH*grid.sinTH

urr[grid.RR < rb] = 0
uth[grid.RR < rb] = 0

#θ＝０(回転軸)(対称性)
# 境界の外で子午面流の設定
for i in range(0,grid.margin):
   # upper
   urr[i,:] = - urr[2*grid.margin-i-1,:]
   uth[i,:] = + uth[2*grid.margin-i-1,:]
   
   # lower
   urr[grid.ixg-i-1,:] = - urr[grid.ixg-2*grid.margin+i,:]
   uth[grid.ixg-i-1,:] = + uth[grid.ixg-2*grid.margin+i,:]

# latitudinal boundary
for j in range(0,grid.margin):
   # pole
   urr[:,j] = + urr[:,2*grid.margin - j - 1] # symmetric
   uth[:,j] = - uth[:,2*grid.margin - j - 1] # antisymetric

   # equator
   urr[:,grid.jxg-j-1] = + urr[:,grid.jxg-2*grid.margin + j] # symmetric
   uth[:,grid.jxg-j-1] = - uth[:,grid.jxg-2*grid.margin + j] # antisymmetric

#CFL condition
c_cfl=0.1
dtmin = 1.e10
for i in range(grid.margin,grid.ixg-grid.margin):
   for j in range(grid.margin,grid.jxg-grid.margin):
      dt_adv = c_cfl * np.min([grid.drr, grid.rr[i]*grid.dth])/ np.sqrt(urr[i,j]**2 + uth[i,j]**2)
      dt_dif = c_cfl * np.min([grid.drr, grid.rr[i]*grid.dth])**2 /(2 * et[i,j])
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
   Bph[0:ibase,:] = 0

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
      ax.pcolormesh(grid.Y,grid.X,Bph,vmax=1.e0,vmin=-1.e0,cmap='bwr')
      ax.contour(grid.Y,grid.X,grid.RR/cfg.RSUN*grid.sinTH*Aph,colors='black',levels=np.linspace(-8.e12,8.e12,10))
      ax.set_xlim(0,cfg.RSUN)
      ax.set_ylim(-cfg.RSUN,cfg.RSUN)
      plt.pause(0.01)
      print(time/86400,n,nd)
      Brr =  dth2(grid.sinTH*Aph,grid.dth)/grid.RR/grid.sinTH
      Bth = -drr2(   grid.RR*Aph,grid.drr)/grid.RR
      np.savez(file='data/data.'+str(nd).zfill(6)+'.npz' \
            ,Aph=Aph,Bph=Bph,Brr=Brr,Bth=Bth,time=time)
                
   ####ダイナモ方程式                       
   Bphm, Aphm = time_marching(Bph , Aph ,dt, urr, uth, grid, et, etrr, so, omrr, omth, ibase)
   
   Aphm, Bphm = boundary_condition(Aphm, Bphm, grid.margin, grid.rr, grid.th, grid.ixg, grid.jxg)

   Bphn, Aphn = time_marching(Bphm, Aphm, dt, urr, uth, grid, et, etrr, so, omrr, omth, ibase)
   Aphn, Bphn = boundary_condition(Aphn, Bphn,grid.margin,grid.rr,grid.th,grid.ixg,grid.jxg)
    
   Bph = 0.5*Bph + 0.5*Bphn
   Aph = 0.5*Aph + 0.5*Aphn

   