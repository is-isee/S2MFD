
import numpy as np
from FLD_ISEE.tools import drr1, drr2, dth1, dth2
from numba import njit

@njit
def poloidal_mag(Aph, RR, sinTH, drr, dth):
   # poloidal magnetic field
   Brr = + dth2(sinTH*Aph,dth)/RR/sinTH
   Bth = - drr2(   RR*Aph,drr)/RR
   
   return Brr, Bth

@njit
def advection(Bph, Aph, RR, sinTH, urr,uth,drr,dth):
   # 磁場の微分(移流量)
   Bph_adrr = - drr2(RR*urr*Bph,drr)/RR #  動径方向の移流(Bph)
   Bph_adth = - dth2(   uth*Bph,dth)/RR #  緯度方向の移流(Bph)
   
   Aph_adrr = - drr2(   RR*Aph,drr)*urr/RR       # 動径方向の移流(Aph)
   Aph_adth = - dth2(sinTH*Aph,dth)*uth/RR/sinTH # 緯度方向の移流(Aph)

   return Bph_adrr, Bph_adth, Aph_adrr, Aph_adth

@njit
def diffusion(Bph, Aph, RR, sinTH, drr, dth, et, etrr):
   # magnetic derivative
   Bphrr = drr1(Bph,drr,'up')
   Bphth = dth1(Bph,dth,'up')
   Aphrr = drr1(Aph,drr,'up')
   Aphth = dth1(Aph,dth,'up')
   
   Bph_dfrr = + et*drr1(RR**2*Bphrr,drr,'dw')/RR**2
   Bph_dfth = + et*dth1(sinTH*Bphth,dth,'dw')/RR**2/sinTH
   Bph_dfex = - et*Bph/RR**2/sinTH**2
   
   Aph_dfrr = + et*drr1(RR**2*Aphrr,drr,'dw')/RR**2
   Aph_dfth = + et*dth1(sinTH*Aphth,dth,'dw')/RR**2/sinTH
   Aph_dfex = - et*Aph/RR**2/sinTH**2
   
   # diffusivity gradient influence
   Bph_dfrrg = etrr*drr1(RR*Bph,drr,'dw')/RR
   
   return Bph_dfrr, Bph_dfth, Bph_dfex, Bph_dfrrg, Aph_dfrr, Aph_dfth, Aph_dfex

@njit
def omega_effect(Brr, Bth, RR, sinTH, omrr, omth):
    Bph_omrr = Brr*omrr*RR*sinTH
    Bph_omth = Bth*omth*RR*sinTH
    
    return Bph_omrr, Bph_omth

def alpha_effect(Bph, Aph, rr, ibase, so):
    Bphso = np.repeat(Bph[ibase, :][np.newaxis, :], len(rr), axis=0)
    #tmp, Bphso = np.meshgrid(rr,Bph[ibase,:], indexing='ij')
    Aph_sour = so*Bphso/(1 + (Bphso)**2)
        
    return Aph_sour

def time_marching(Bph, Aph, dt, grid, setup): 
   # calculate poloidal magnetic field
   Brr, Bth = poloidal_mag(Aph, grid.RR, grid.sinTH, grid.drr, grid.dth)
   
   # advection term
   Bph_adrr, Bph_adth, Aph_adrr, Aph_adth \
        = advection(Bph, Aph, grid.RR, grid.sinTH, setup.urr, setup.uth, grid.drr, grid.dth)
        
   # diffusion term
   Bph_dfrr, Bph_dfth, Bph_dfex, Bph_dfrrg, Aph_dfrr, Aph_dfth, Aph_dfex \
          = diffusion(Bph, Aph, grid.RR, grid.sinTH, grid.drr, grid.dth, setup.et, setup.etrr)   
   # Omega effect
   Bph_omrr, Bph_omth = omega_effect(Brr, Bth, grid.RR, grid.sinTH, setup.omrr, setup.omth)
   
   # source term
   Aph_sour = alpha_effect(Bph, Aph, grid.rr, setup.ibase, setup.so)
      
   dBph = + (Bph_adrr + Bph_adth) \
          + (Bph_dfrr + Bph_dfth + Bph_dfrrg + Bph_dfex) \
          + (Bph_omrr + Bph_omth) 
          
   dAph = + (Aph_adrr + Aph_adth) \
          + (Aph_dfrr + Aph_dfth + Aph_dfex) \
          + Aph_sour
      
   Bphm = Bph + dt*dBph
   Aphm = Aph + dt*dAph
   
   return Bphm, Aphm

