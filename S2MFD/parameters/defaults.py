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

# 動径運動量に磁気圧勾配 (磁気浮力) を含めるか。False にすると
# Rempel (2006) の "magnetic buoyancy off" 解になる (表1 列 4/6/8)。
magnetic_buoyancy = True

# --- 動径方向の非一様格子 -------------------------------------------------
# 0 で一様 (既定、従来と厳密に同一)。>0 で grid_stretch_center 付近に
# 点を集める (sinh 写像)。対流層/放射層境界のように、領域幅に対して薄い
# 境界層が解を支配する問題で効く。b=2.5 でタコクライン (0.05 R_sun) の
# セル数が 17 -> 27 になり、表面は 2.2 倍粗くなる。
# 上下境界で v_r ではなく質量フラックス rho_0 r^2 v_r を反対称にする。
# False にすると旧来の v_r 反対称 (比較用)。
mass_flux_bc = True

grid_stretch = 0.0
grid_stretch_center = 0.0
grid_stretch_width = 0.0   # 0 で自動 (領域幅の 15%)

# Flag for continuation
cont_flag = True

# fixed parameters (modifications are not recommended)
datadir = 'data/'
gridfile = 'grid.npz'
setupfile = 'setup.npz'
legendrefile = 'legendre.npz'
configfile = 'config.json'

# --- 下部境界だけ磁場の人工拡散を効かせる (既定は無効) ----------------------
# 磁場フィルタの特性速度は |v| だけなので、子午面流が入り込まない放射層では
# 実質効かない。Rempel (2006) どおりの下部境界 B_Phi = 0 を課すときだけ必要。
# 詳細は S2MFD/physics/dynamic.py の DynamicSolver.__init__ を参照。
sld_bottom_speed = 0.0      # [cm/s] 特性速度の床。0 で無効
sld_bottom_width = 0.0      # [cm] 下端からの幅。0 なら 0.02*RSUN を使う

# 磁場の人工拡散フラックスを動径境界で開ける。反対称境界 (B_Phi = 0) を
# 使うときだけ必要。対称境界では何も変わらない。
magnetic_open_boundary = False
