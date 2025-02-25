from S2MFD.parameters.defaults import *

meridional_circulation_type = 'D99'
uu0 = 1000
m   = 0.5
p   = 0.25
q   = 0
rrb = 0.71*RSUN # base of the meridional flow

xi0 = RSUN/rrb - 1
c1d  = (2*m+1)*(m+p)/(m+1)/p * (xi0**(-m))
c2d  = (2*m+p+1)*m/(m+1)/p * (xi0**(-(m+p)))