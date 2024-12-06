
import numpy as np
from S2MFD.tools import drr1, drr2, dth1, dth2
from numba import njit

@njit
def poloidal_mag(Aph, RR, sinTH, drr, dth):
   """
   Calculate the poloidal magnetic field.

   Parameters
   ----------
   Aph : numpy.ndarray
      Longitudinal vector potential
   RR : numpy.ndarray
      Radial coordinate
   sinTH : numpy.ndarray
   drr : numpy.ndarray
      Radial grid spacing
   dth : numpy.ndarray
      Colatitudinal grid spacing

   Returns
   -------
   tuple of numpy.ndarray, float
      - Brr : Radial  magnetic field
      - Bth : Latitudinal magnetic field
         
   """
   Brr = + dth2(sinTH*Aph,dth)/RR/sinTH
   Bth = - drr2(   RR*Aph,drr)/RR
   
   return Brr, Bth

@njit
def advection(Bph, Aph, RR, sinTH, urr,uth,drr,dth):
   """
   Calculate the advection terms of the magnetic field.

   Parameters
   ----------
   Bph : numpy.ndarray
      Longitudinal magnetic field
   Aph : numpy.ndarray
      Longitudinal vector potential
   RR : numpy.ndarray
      Radial coordinate
   sinTH : numpy.ndarray
   urr : numpy.ndarray
      Radial velocity
   uth : numpy.ndarray
      Colatitudinal velocity
   drr : numpy.ndarray
      Radial grid spacing
   dth : numpy.ndarray
      Colatitudinal grid spacing

   Returns
   -------
   tuple of numpy.ndarray
      - Bph_adrr : Radial advection of Bph
      - Bph_adth : Colatitudinal advection of Bph
      - Aph_adrr : Radial advection of Aph
      - Aph_adth : Colatitudinal advection of Aph
   """
   # 磁場の微分(移流量)
   # Bph_adrr = - drr2(Bph/RR   ,drr)*urr*RR #  動径方向の移流(Bph)
   # Bph_adth = - dth2(Bph/sinTH,dth)*uth*sinTH/RR #  緯度方向の移流(Bph)

   Bph_adrr = - drr2(Bph*urr*RR,drr)/RR #  動径方向の移流(Bph)
   Bph_adth = - dth2(Bph*uth   ,dth)/RR #  緯度方向の移流(Bph)
   
   Aph_adrr = - drr2(Aph*RR   ,drr)*urr/RR       # 動径方向の移流(Aph)
   Aph_adth = - dth2(Aph*sinTH,dth)*uth/sinTH/RR # 緯度方向の移流(Aph)

   return Bph_adrr, Bph_adth, Aph_adrr, Aph_adth

@njit
def diffusion(Bph, Aph, RR, sinTH, RRm, sinTHm, drr, dth, et, etrr):
   """
   Calculate the diffusion terms of the magnetic field.

   Parameters
   ----------
   Bph : numpy.ndarray
      Longitudinal magnetic field
   Aph : numpy.ndarray
      Longitudinal vector potential
   RR : numpy.ndarray
      Radial coordinate
   sinTH : numpy.ndarray
   drr : numpy.ndarray
      Radial grid spacing
   dth : numpy.ndarray
      Colatitudinal grid spacing
   et : numpy.ndarray
      magnetic diffusivity
   etrr : numpy.ndarray
      Radial gradient of magnetic diffusivity
   
   Returns
   -------
   tuple of numpy.ndarray
      - Bph_dfrr : Radial diffusion of Bph
      - Bph_dfth : Colatitudinal diffusion of Bph
      - Bph_dfex : Extra diffusion term of Bph
      - Bph_dfrrg : Radial gradient diffusion of Bph
      - Aph_dfrr : Radial diffusion of Aph
      - Aph_dfth : Colatitudinal diffusion of Aph
      - Aph_dfex : Extra diffusion term of Aph
      
   """   
   # magnetic derivative
   Bphrr = drr1(Bph,drr,'up')
   Bphth = dth1(Bph,dth,'up')
   Aphrr = drr1(Aph,drr,'up')
   Aphth = dth1(Aph,dth,'up')
   
   Bph_dfrr = + et*drr1(RRm**2*Bphrr,drr,'dw')/RR**2
   Bph_dfth = + et*dth1(sinTHm*Bphth,dth,'dw')/RR**2/sinTH
   Bph_dfex = - et*Bph/RR**2/sinTH**2
   
   Aph_dfrr = + et*drr1(RRm**2*Aphrr,drr,'dw')/RR**2
   Aph_dfth = + et*dth1(sinTHm*Aphth,dth,'dw')/RR**2/sinTH
   Aph_dfex = - et*Aph/RR**2/sinTH**2
   
   # diffusivity gradient influence
   Bph_dfrrg = etrr*drr2(RR*Bph,drr)/RR
   
   return Bph_dfrr, Bph_dfth, Bph_dfex, Bph_dfrrg, Aph_dfrr, Aph_dfth, Aph_dfex

@njit
def omega_effect(Brr, Bth, RR, sinTH, omrr, omth):
   """
   Calculate the omega effect.

   Parameters
   ----------
   Brr : numpy.ndarray
      Radial magnetic field
   Bth : numpy.ndarray
      Colatitudinal magnetic field
   RR : numpy.ndarray
      Radial coordinate
   sinTH : numpy.ndarray
   omrr : numpy.ndarray
      Radial gradient of angular velocity
   omth : numpy.ndarray
      Colatitudinal gradient of angular velocity

   Returns
   -------
   tuple of numpy.ndarray
   Bph_omrr : Radial omega effect term of Bph
   Bph_omrh : Colatitudinal omega effect term of Bph
   
   """
   Bph_omrr = Brr*omrr*RR*sinTH
   Bph_omth = Bth*omth*RR*sinTH
   
   return Bph_omrr, Bph_omth

def alpha_effect(Bph, Aph, rr, ibase, so, alpha_type):
   """
   Calculate the alpha effect.

   Parameters
   ----------
   Bph : numpy.ndarray
      Longitudinal magnetic field
   Aph : numpy.ndarray
      Longitudinal vector potential
   rr : numpy.ndarray
      Radial coordinate
   ibase : int
      Radial index for the base of the convection zone
   so : float
      Source term coefficient
   Returns
   -------
   Aph_sour : numpy.ndarray
      Source term due to alpha effect
   """
   if alpha_type == 'BL':
      Bphso = np.repeat(Bph[ibase, :][np.newaxis, :], len(rr), axis=0)
      Aph_sour = so*Bphso/(1 + (Bphso)**2)
   elif alpha_type == 'normal':
      Aph_sour = so*Bph/(1 + (Bph)**2)
      
   return Aph_sour

def time_marching(Bph, Aph, dt, cfg, grid, setup):
   """
   Perform time marching for the magnetic field.

   Parameters
   ----------
   Bph : numpy.ndarray  
      Longitudinal magnetic field
   Aph : numpy.ndarray
      Longitudinal vector potential
   dt : float
      Time step
   grid : object
      Object containing grid information
   setup : object
      Object containing setup information
   Returns
   -------
   tuple of numpy
      - Bphm : Updated longitudinal magnetic field
      - Aphm : Updated longitudinal vector potential
   """   
   # calculate poloidal magnetic field
   Brr, Bth = poloidal_mag(Aph, grid.RR, grid.sinTH, grid.drr, grid.dth)
   
   # advection term
   Bph_adrr, Bph_adth, Aph_adrr, Aph_adth \
        = advection(Bph, Aph, grid.RR, grid.sinTH, setup.urr, setup.uth, grid.drr, grid.dth)
        
   # diffusion term
   Bph_dfrr, Bph_dfth, Bph_dfex, Bph_dfrrg, Aph_dfrr, Aph_dfth, Aph_dfex \
          = diffusion(Bph, Aph, grid.RR, grid.sinTH, grid.RRm, grid.sinTHm, grid.drr, grid.dth, setup.et, setup.etrr)   
   # Omega effect
   Bph_omrr, Bph_omth = omega_effect(Brr, Bth, grid.RR, grid.sinTH, setup.omrr, setup.omth)
   
   # source term
   Aph_sour = alpha_effect(Bph, Aph, grid.rr, setup.ibase, setup.so, cfg.alpha_type)
      
   dBph = + (Bph_adrr + Bph_adth) \
          + (Bph_dfrr + Bph_dfth + Bph_dfrrg + Bph_dfex) \
          + (Bph_omrr + Bph_omth) 
          
   dAph = + (Aph_adrr + Aph_adth) \
          + (Aph_dfrr + Aph_dfth + Aph_dfex) \
          + Aph_sour
      
   Bphm = Bph + dt*dBph
   Aphm = Aph + dt*dAph
   
   return Bphm, Aphm
