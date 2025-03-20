from S2MFD.parameters.defaults import *

# geometry parammeters
rrmin = 0.60*RSUN

# boundary condition
boundary_condition_type = 'potential'

# Diffusivity
# TODO etsはさまざまな値を検証しているので可変的
diffusive_type = 'H10'
etc = 5.e8
ett = 5.e10
ets = 9.e12
dh1 = 0.05*RSUN
dh2 = 0.05*RSUN

## Differential rotation
differential_type = 'H10'
ome =  460.7e-9*2*np.pi # rotation rate at equator
omc =  432.8e-9*2*np.pi # rotation rate at radiative zone
a2  = -62.69e-9*2*np.pi 
a4  = -67.13e-9*2*np.pi 

## Alpha effect
# 北南ともに計算しているので少しプロファイルが変
alpha_type = 'H10'
so1 = 100 # alpha effect amplitude
r4  = 0.95*RSUN
r5  = RSUN
dh4 = 0.05*RSUN
dh5 = 0.01*RSUN
gam = 30

# meridional flow
meridional_circulation_type = 'H10'
uu0 = 1000
m   = 0.5
p   = 0.25
q   = 0
rrb = 0.62*RSUN # base of the meridional flow

xi0 = RSUN/rrb - 1
c1d  = (2*m+1)*(m+p)/(m+1)/p * (xi0**(-m))
c2d  = (2*m+p+1)*m/(m+1)/p * (xi0**(-(m+p)))