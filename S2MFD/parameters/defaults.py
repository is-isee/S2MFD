# This is the default parameter set.
# Modifications are not recommended.
# TODO デフォルトのパラメタ設定を行うファイル。
# TODO Jouve＋2008と照らし合わせて確認。
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

# Setup parameters
## geometry parameters
rrc = 0.7*RSUN  # base of the convection zone
d   = 0.02*RSUN # width of the tachocline

## Diffusivity
etc = 1.e9
ett = 1.e11

## Differential rotation
com = 1.4e5
#ome = 456.e-9*2*np.pi # rotation rate at equator
ome = com/RSUN**2*ett
omc = 0.92*ome        # rotation rate at radiative zone
c2 = 0.2*ome          # latitudinal gradient of differential rotation

# TODO ここのcso=35でsample.pyとかのcso=3.5なのはOKか。
# TODO こことα効果の設定をしているファイルが連携していると思うが、どのように連携しているのか。
## Alpha effect
alpha_type = 'BL'
cso = 35 # alpha non-dimensional parameter
so0 = cso*ett/RSUN # alpha effect amplitude
r1  = 0.95*RSUN # bottom of alpha effect
d1  = 0.01*RSUN # width of alpha effect

## Meridional circulation
rey = 700
#uu0 = 1000 # flow amplitude
uu0 = rey*ett/RSUN
rrb = 0.65*RSUN # base of the meridional flow

# Flag for continuation
cont_flag = True

# fixed parameters (modifications are not recommended)
datadir = 'data/'
gridfile = 'grid.npz'
setupfile = 'setup.npz'
configfile = 'config.json'