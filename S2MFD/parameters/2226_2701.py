from S2MFD.parameters.defaults import *

tend = 91000*d2s
# boundary condition
boundary_condition_type = 'potential'

def uu0_time_dependent(a0_u,a1_u,a2_u,a3_u,time):
    omega_u = 2*np.pi/(52*365*60*60*24)
    u0t = a0_u + a1_u*np.sin(1.0*omega_u*time) + a2_u*np.sin(2.0*omega_u*time) + a3_u*np.sin(3.0*omega_u*time)
    return u0t

# def so0_time_dependent(a0_s,a1_s,a2_s,a3_s,time):
#     omega_s = 2*np.pi/(52*365*60*60*24)
#     s0t = a0_s + a1_s*np.sin(1.0*omega_s*time) + a2_s*np.sin(2.0*omega_s*time) + a3_s*np.sin(3.0*omega_s*time)
#     return s0t

# def uu0_known(time):
#     omega_u = 2*np.pi/(52*365*60*60*24)
#     a0_u=791.013294
#     a1_u=73.513338
#     a2_u=-138.214256
#     a3_u=33.452918
#     u0t = a0_u + a1_u*np.sin(1.0*omega_u*time) + a2_u*np.sin(2.0*omega_u*time) + a3_u*np.sin(3.0*omega_u*time)
#     return u0t 
# uu0_const = 791.013294