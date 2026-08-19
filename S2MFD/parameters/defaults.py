# This is the default parameter set.
# Modifications are not recommended.

import numpy as np

# fixed parameters (modifications are not recommended)
margin = 1
RSUN = 6.96e10

# time parameters
d2s = 86400 # day to second
tend = 200000*d2s
dtout = 40*d2s

# geometry parameters
ix = 128 # number of grid points in r-direction
jx = 128 # number of grid points in theta-direction
rrmin = 0.65*RSUN
rrmax = RSUN
thmin = 0
thmax = np.pi

# boundary condition
boundary_condition_type = 'vertical'

# Setup parameters
## geometry parameters
rrc = 0.7*RSUN  # base of the convection zone
d   = 0.02*RSUN # width of the tachocline

## Diffusivity
diffusive_type = 'J08'
etc = 1.e9
ett = 1.e11

## Differential rotation
differential_type = 'J08'
com = 1.4e5
#ome = 456.e-9*2*np.pi # rotation rate at equator
ome = com/RSUN**2*ett
omc = 0.92*ome        # rotation rate at radiative zone
c2 = 0.2*ome          # latitudinal gradient of differential rotation


## Alpha effect
alpha_type = 'BL'
cso = 35 # alpha non-dimensional parameter
so0 = cso*ett/RSUN # alpha effect amplitude
r1  = 0.95*RSUN # bottom of alpha effect
d1  = 0.01*RSUN # width of alpha effect

## Meridional circulation
meridional_circulation_type = 'J08'
rey = 700
#uu0 = 1000 # flow amplitude
uu0 = rey*ett/RSUN
rrb = 0.65*RSUN # base of the meridional flow

# Dynamics mode
#   'kinematic' : 流れ場を与えて誘導方程式だけを解く (既定。従来の S2MFD)
#   'dynamic'   : 流体の運動方程式とエントロピー方程式も解く (Rempel 2006)
#   'hydro'     : 磁場を解かず流体だけを解く (差動回転の緩和用)
# 'kinematic' 以外では以下の熱力学パラメタが必要になる。
dynamics = 'kinematic'

# Flag for continuation
cont_flag = True

# fixed parameters (modifications are not recommended)
datadir = 'data/'
gridfile = 'grid.npz'
setupfile = 'setup.npz'
legendrefile = 'legendre.npz'
configfile = 'config.json'