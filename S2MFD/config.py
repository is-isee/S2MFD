import numpy as np
# fixed parameters (modifications are not recommended)
margin = 1
RSUN = 6.96e10

# time parameter
d2s = 86400 # day to second
tend = 30000*d2s
dtout = 100*d2s

# geometry parameters
ix = 128 # number of grid points in r-direction
jx = 128 # number of grid points in theta-direction
rrmin = 0.65*RSUN
rrmax = RSUN
thmin = 0
thmax = np.pi

# Setup parameters
## geometry parameters
rrc = 0.7*RSUN  # base of the convection zone
d   = 0.02*RSUN # width of the tachocline

## Differential rotation
ome = 456.e-9*2*np.pi # rotation rate at equator
omc = 0.92*ome        # rotation rate at radiative zone
c2 = 0.2*ome          # latitudinal gradient of differential rotation

## Diffusivity
etc = 1.e9
ett = 1.e11

## Alpha effect
cso = 35 # alpha non-dimensional parameter
so0 = cso*ett/RSUN # alpha effect amplitude
r1  = 0.95*RSUN # bottom of alpha effect
d1  = 0.05*RSUN # width of alpha effect

## Meridional circulation
uu0 = 1000 # flow amplitude
rrb = 0.65*RSUN # base of the meridional flow

# Flag for continuation
cont_flag = True

# fixed parameters (modifications are not recommended)
datadir = 'data/'
gridfile = datadir+'grid.npz'
setupfile = datadir+'setup.npz'