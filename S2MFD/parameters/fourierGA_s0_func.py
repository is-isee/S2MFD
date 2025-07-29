from S2MFD.parameters.defaults import *

tend = 91000*d2s
# boundary condition
boundary_condition_type = 'potential'

def so0_time_dependent(a0_s,a1_s,a2_s,b1_s,b2_s,omega_s,time):
    s0t = (a0_s   + a1_s*np.cos(1.0*omega_s*time) + b1_s*np.sin(1.0*omega_s*time)\
                  + a2_s*np.cos(2.0*omega_s*time) + b2_s*np.sin(2.0*omega_s*time)) * ett / RSUN
    return s0t
