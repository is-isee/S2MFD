from S2MFD.parameters.defaults import *

import numpy as np



# time parameters
dtout = 30*d2s

# geometry parameters
ix = 129 # number of grid points in r-direction
jx = 129 # number of grid points in theta-direction
rrmin = 0.55*RSUN


# boundary condition
boundary_condition_type = 'potential'

# Setup parameters
## geometry parameters
rrc = 0.7*RSUN  # base of the convection zone
d   = 0.05*RSUN # width of the tachocline

## Diffusivity
# 最後にdefault.pyを変更する必要あり。
diffusive_type = "C04"
etR = 2.2e8
etS = 2.4e12

## Differential rotation
differential_type = "C04"
omR =  432.8e-9*2*np.pi
ome =  460.7e-9*2*np.pi
a2  = -62.69e-9*2*np.pi
a4  = -67.13e-9*2*np.pi


## Alpha effect
alpha_type = 'C04'
so0 = 2500 # alpha effect amplitude(論文の値が少し怪しい)
r1  = 0.95*RSUN # bottom of alpha effect
d1  = 0.025*RSUN # width of alpha effect

## Meridional circulation
meridional_circulation_type = 'J08'
rey = 700
#uu0 = 1000 # flow amplitude
uu0 = rey*ett/RSUN
rrb = 0.65*RSUN # base of the meridional flow
