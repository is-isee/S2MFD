from dataclasses import dataclass, field
from FLD_ISEE.tools import drr2, dth2
import FLD_ISEE.config as cfg
import numpy as np
import pickle
from scipy.special import erf

@dataclass
class setup_c:
   """
   A class to configure and initialize physical properties and flow patterns in a spherical 
   coordinate grid for simulations. It supports the setup of differential rotation, 
   diffusivity, alpha effect, and meridional flow based on input configurations and grid.

   Attributes:
      urr (np.ndarray): Radial component of the meridional flow velocity.
      uth (np.ndarray): Latitudinal component of the meridional flow velocity.
      om (np.ndarray): Angular velocity distribution based on the differential rotation profile.
      omrr (np.ndarray): Radial derivative of angular velocity.
      omth (np.ndarray): Latitudinal derivative of angular velocity.
      et (np.ndarray): Magnetic diffusivity profile.
      etrr (np.ndarray): Radial derivative of the magnetic diffusivity.
      ibase (int): Index corresponding to the tachocline region in the radial direction.

   Methods:
      __init__(cfg, grid):
         Initializes the setup using the provided configuration and grid properties.
      
      save(filename):
         Saves the instance of the class to a file using pickle.

      load(filename):
         Loads a previously saved instance of the class from a file.

   Initialization Details:
      The class calculates various properties based on the input configuration (`cfg`) and 
      grid (`grid`). Key processes include:
      
      - Differential rotation (`om`): Calculated using an error function-based profile to 
         model the transition between regions of different angular velocities.
      - Magnetic diffusivity (`et`): A radial profile with a smooth transition across 
         specified regions.
      - Meridional flow (`urr` and `uth`): Modeled based on analytical expressions with 
         boundary corrections to ensure symmetry at the poles and within the computational margins.
      - Alpha effect (`so`): Derived based on the cosine and sine of the colatitude (`cosTH` 
         and `sinTH`) for dynamo modeling.
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