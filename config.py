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

# time parameter
d2s = 86400 # day to second
tend = 30000*d2s
dtout = 100*d2s

# Setup parameters
## Differential rotation

## Diffusivity

## Alpha effect

## Meridional circulation

# fixed parameters (modifications are not recommended)
datadir = 'data/'
gridfile = datadir+'grid.pkl'