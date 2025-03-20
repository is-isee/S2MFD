from dataclasses import dataclass, field
from S2MFD.tools import drr2, dth2
import S2MFD.cfg as Cfg
import numpy as np
import pickle
from scipy.special import erf

@dataclass
class Setup:
   """
   Class to configure and initialize physical properties used in simulations.

   Attributes
   ----------
   urr : numpy.ndarray
      Radial component of the meridional flow velocity.
   uth : numpy.ndarray
      Latitudinal component of the meridional flow velocity.
   om : numpy.ndarray
      Angular velocity distribution based on the differential rotation profile.
   omrr : numpy.ndarray
      Radial derivative of angular velocity.
   omth : numpy.ndarray
      Latitudinal derivative of angular velocity.
   et : numpy.ndarray
      Magnetic diffusivity profile.
   etrr : numpy.ndarray
      Radial derivative of the magnetic diffusivity.
   ibase : int
      Index corresponding to the tachocline region in the radial direction.

   """
   urr: np.ndarray = field(init=False)
   uth: np.ndarray = field(init=False)   
   om: np.ndarray = field(init=False)   
   omrr: np.ndarray = field(init=False)
   omth: np.ndarray = field(init=False)
   et: np.ndarray = field(init=False)
   etrr: np.ndarray = field(init=False)
   ibase: int = field(init=False)
   
   def __init__(self,cfg,grid):
      """
      Initialize the setup object.
      
      Parameters
      ----------
      cfg : S2MFD.Cfg
         Configuration object.
         
      grid : S2MFD.Grid
         Grid object.
      """
      # differential rotation
      if cfg.differential_type == 'J08':
         self.om = cfg.omc + 0.5*(1 + erf((grid.RR-cfg.rrc)/cfg.d))*(cfg.ome - cfg.omc - cfg.c2*grid.cosTH**2)
      elif cfg.differential_type == 'H10':
         self.om = cfg.omc + 0.5*(1 + erf(2*(grid.RR-cfg.rrc)/cfg.dh1))*(cfg.ome + cfg.a2*grid.cosTH**2 + cfg.a4*grid.cosTH**4 - cfg.omc)
      self.omrr = drr2(self.om, grid.drr)
      self.omth = dth2(self.om, grid.dth)/grid.RR
   
      # diffusivity
      if cfg.diffusive_type == 'J08':
         self.et = cfg.etc + 0.5*(cfg.ett - cfg.etc)*(1 + erf((grid.RR-cfg.rrc)/cfg.d))
      if cfg.diffusive_type == 'H10':
         self.et = cfg.etc + 0.5*cfg.ett*(1 + erf((grid.RR-cfg.rrc)/cfg.dh1)) + 0.5*cfg.ets*(1 + erf((grid.RR-cfg.r1)/cfg.dh2))
      self.etrr = drr2(self.et, grid.drr)

      #タコクラインのindex
      self.ibase = np.argmin(abs(grid.rr - cfg.rrc))

      # alpha effect
      if cfg.alpha_type == 'BL':
         self.so = cfg.so0*0.5 \
            *(1 + erf((grid.RR-cfg.r1)/cfg.d1))*(1 - erf((grid.RR-cfg.RSUN)/cfg.d1)) \
               *grid.cosTH*grid.sinTH
      elif cfg.alpha_type == 'normal':
         self.so = cfg.so0*3*np.sqrt(3)/4 \
            *(1 + erf((grid.RR-cfg.rrc)/cfg.d)) \
               *grid.sinTH**2*grid.cosTH
      elif cfg.alpha_type == 'H10':
         self.so = cfg.so1*0.25 \
            *(1+erf((grid.RR-cfg.r4)/cfg.dh4))*(1-erf((grid.RR-cfg.r5)/cfg.dh5)) \
               *grid.cosTH*grid.sinTH*(1/(1+np.e**(-cfg.gam*(grid.TH[1,:]-np.pi*0.25)))+1/(1+np.e**(-cfg.gam*(-grid.TH[1,:]+np.pi*0.75)))-1)
            
      # Meridional flow
      # Meridional flow (Jouve+2008 Model)
      if cfg.meridional_circulation_type == 'J08':
         self.urr = -cfg.uu0*2*(cfg.RSUN - cfg.rrb)/np.pi/grid.RR \
            *(grid.RR-cfg.rrb)**2/(cfg.RSUN - cfg.rrb)**2 \
            *np.sin(np.pi*(grid.RR-cfg.rrb)/(cfg.RSUN-cfg.rrb))*(3*grid.cosTH**2 - 1)
            
         self.uth = cfg.uu0*((3*grid.RR-cfg.rrb)/(cfg.RSUN-cfg.rrb) \
               *np.sin(np.pi*(grid.RR-cfg.rrb)/(cfg.RSUN-cfg.rrb)) \
               + grid.RR*np.pi/(cfg.RSUN-cfg.rrb)*(grid.RR-cfg.rrb)/(cfg.RSUN-cfg.rrb) \
                  *np.cos(np.pi*(grid.RR-cfg.rrb)/(cfg.RSUN-cfg.rrb))) \
               *2*(cfg.RSUN-cfg.rrb)/np.pi/grid.RR*(grid.RR-cfg.rrb)/(cfg.RSUN-cfg.rrb) \
                  *grid.cosTH*grid.sinTH
      # Meridional flow (Dikpati+1999 Model)
      elif cfg.meridional_circulation_type == 'D99':
         xi  = cfg.RSUN/grid.RR  - 1
         xi[grid.RR > cfg.RSUN] = 0 
         
         self.urr = cfg.uu0*(cfg.RSUN/grid.RR) \
            *(-1/(cfg.m+1) + cfg.c1d/(2*cfg.m + 1)*xi**cfg.m - cfg.c2d/(2*cfg.m+cfg.p+1)*xi**(cfg.m+cfg.p)) \
            *xi*grid.sinTH**cfg.q*( (cfg.q+2)*grid.cosTH**2 - grid.sinTH**2)

         self.uth = cfg.uu0*((cfg.RSUN/grid.RR)**3) \
            *(-1+cfg.c1d*xi**cfg.m - cfg.c2d*xi**(cfg.m+cfg.p)) \
            *grid.sinTH**(cfg.q+1)*grid.cosTH
      
      elif cfg.meridional_circulation_type == 'H10':
         xi  = cfg.RSUN/grid.RR  - 1
         xi[grid.RR > cfg.RSUN] = 0
         
         self.urr = cfg.uu0*(cfg.RSUN/grid.RR)**2 \
            *(-1/(cfg.m+1) + cfg.c1d/(2*cfg.m + 1)*xi**cfg.m - cfg.c2d/(2*cfg.m+cfg.p+1)*xi**(cfg.m+cfg.p)) \
            *xi*grid.sinTH**cfg.q*((cfg.q+2)*grid.cosTH**2 - grid.sinTH**2)

         self.uth = cfg.uu0*((cfg.RSUN/grid.RR)**3) \
            *(-1+cfg.c1d*xi**cfg.m - cfg.c2d*xi**(cfg.m+cfg.p)) \
            *grid.sinTH**(cfg.q+1)*grid.cosTH
            
      self.urr[grid.RR < cfg.rrb] = 0
      self.uth[grid.RR < cfg.rrb] = 0
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
      """
      Save the setup data to a file.
      
      Parameters
      ----------
      filename : str
         File path to save the setup data.
      """
      np.savez(filename, **self.__dict__)
   
   @classmethod
   def load(cls, filename):
      """
      Load the setup data from a file.
      
      Parameters
      ----------
      filename : str
         File path to load the setup data.
      """
      data = np.load(filename,allow_pickle=True)
      obj = cls.__new__(cls)
      obj.__dict__.update({key: data[key] for key in data.files})
      return obj