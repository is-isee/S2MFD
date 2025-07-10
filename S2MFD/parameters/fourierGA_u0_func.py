from S2MFD.parameters.defaults import *

tend = 91000*d2s
# boundary condition
boundary_condition_type = 'potential'

def uu0_time_dependent(a0_u,a1_u,a2_u,b1_u,b2_u,omega_u,time):
    u0t = (a0_u   + a1_u*np.cos(1.0*omega_u*time) + b1_u*np.sin(1.0*omega_u*time)\
                  + a2_u*np.cos(2.0*omega_u*time) + b2_u*np.sin(2.0*omega_u*time)) * ett / RSUN
    return u0t

