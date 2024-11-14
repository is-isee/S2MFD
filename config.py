import numpy as np
# fixed parameters (modifications are not recommended)
margin = 1
RSUN = 6.96e10

# Flag for continuation
cont_flag = True

# geometry parameters
ix = 64
jx = 128
rrmin = 0.65*RSUN
rrmax = RSUN
thmin = 0
thmax = np.pi

# Setup parameters
## Differential rotation

## Diffusivity

## Alpha effect

## Meridional circulation

# fixed parameters (modifications are not recommended)
datadir = 'data/'
gridfile = datadir+'grid.pkl'