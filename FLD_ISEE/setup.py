from dataclasses import dataclass, field
from .tools import drr1, drr2, dth1, dth2
import FLD_ISEE.config as cfg
import numpy as np
import pickle
from scipy.special import erf

@dataclass
class setup_c:
   urr: np.ndarray = field(init=False)
   uth: np.ndarray = field(init=False)   
   om: np.ndarray = field(init=False)   
   omrr: np.ndarray = field(init=False)
   omth: np.ndarray = field(init=False)
   et: np.ndarray = field(init=False)
   etrr: np.ndarray = field(init=False)
   ibase: int = field(init=False)
   
   def __init__(self,cfg,grid):
      # differential rotation
      self.om = cfg.omc + 0.5*(1 + erf((grid.RR-cfg.rrc)/cfg.d))*(cfg.ome - cfg.omc - cfg.c2*grid.cosTH**2)

      self.omrr = drr2(self.om, grid.drr)
      self.omth = dth2(self.om, grid.dth)/grid.RR
   
      # diffusivity
      self.et = cfg.etc + 0.5*(cfg.ett - cfg.etc)*(1 + erf((grid.RR-cfg.rrc)/cfg.d))
      self.etrr = drr2(self.et, grid.drr)

      #タコクラインのindex
      self.ibase = np.argmin(abs(grid.rr - cfg.rrc))

      # alpha effect
      self.so = cfg.so0*0.5 \
         *(1 + erf((grid.RR-cfg.r1)/cfg.d1))*(1 - erf(grid.RR-cfg.RSUN)/cfg.d1) \
            *grid.cosTH*grid.sinTH

      # Meridional flow (Jouve+2008 Model)
      self.urr = -cfg.u0*2*(cfg.RSUN - cfg.rb)/np.pi/grid.RR \
         *(grid.RR-cfg.rb)**2/(cfg.RSUN - cfg.rb)**2 \
         *np.sin(np.pi*(grid.RR-cfg.rb)/(cfg.RSUN-cfg.rb))*(3*grid.cosTH**2 - 1)
         
      self.uth = cfg.u0*((3*grid.RR-cfg.rb)/(cfg.RSUN-cfg.rb) \
            *np.sin(np.pi*(grid.RR-cfg.rb)/(cfg.RSUN-cfg.rb)) \
            + grid.RR*np.pi/(cfg.RSUN-cfg.rb)*(grid.RR-cfg.rb)/(cfg.RSUN-cfg.rb) \
               *np.cos(np.pi*(grid.RR-cfg.rb)/(cfg.RSUN-cfg.rb))) \
            *2*(cfg.RSUN-cfg.rb)/np.pi/grid.RR*(grid.RR-cfg.rb)/(cfg.RSUN-cfg.rb) \
               *grid.cosTH*grid.sinTH

      self.urr[grid.RR < cfg.rb] = 0
      self.uth[grid.RR < cfg.rb] = 0

      #θ＝０(回転軸)(対称性)
      # 境界の外で子午面流の設定
      for i in range(0,grid.margin):
         self.urr[i           ,:] = - self.urr[2*grid.margin-i-1       ,:] # upper
         self.urr[grid.ixg-i-1,:] = - self.urr[grid.ixg-2*grid.margin+i,:] # lower

      # latitudinal boundary
      for j in range(0,grid.margin):
         self.uth[:,j           ] = - self.uth[:,2*grid.margin - j - 1     ] # north pole
         self.uth[:,grid.jxg-j-1] = - self.uth[:,grid.jxg-2*grid.margin + j] # south pole


   def save(self, filename):
      with open(filename, 'wb') as f:
         pickle.dump(self, f)
      print('grid_c instance saved to', filename)
   
   @classmethod
   def load(cls, filename):
      with open(filename, 'rb') as f:
         return pickle.load(f)