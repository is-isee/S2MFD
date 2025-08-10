from S2MFD.parameters.defaults import *

tend = 91000*d2s
# boundary condition
boundary_condition_type = 'potential'

def uu0_time_dependent(a0_u,a1_u,a2_u,a3_u,time):
    omega_u = 2*np.pi/(56*365*60*60*100)
    u0t = a0_u + a1_u*np.sin(1.0*omega_u*time) + a2_u*np.sin(2.0*omega_u*time) + a3_u*np.sin(3.0*omega_u*time)
    return u0t

