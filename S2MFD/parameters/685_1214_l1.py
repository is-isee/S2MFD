from S2MFD.parameters.defaults import *

tend = 91000*d2s
# boundary condition
boundary_condition_type = 'potential'

# def uu0_time_dependent(a0_u,a1_u,a2_u,a3_u,time):
#     omega_u = 2*np.pi/(58*365*60*60*24)
#     u0t = a0_u + a1_u*np.sin(1.0*omega_u*time) + a2_u*np.sin(2.0*omega_u*time) + a3_u*np.sin(3.0*omega_u*time)
#     return u0t

def uu0_time_dependent(a0_u,a1_u,time):
    omega_u = 2*np.pi/(58*365*60*60*24)
    u0t = a0_u + a1_u*np.sin(1.0*omega_u*time)
    return u0t

# def so0_time_dependent(a0_s,a1_s,a2_s,a3_s,time):
#     omega_s = 2*np.pi/(58*365*60*60*24)
#     s0t = a0_s + a1_s*np.sin(1.0*omega_s*time) + a2_s*np.sin(2.0*omega_s*time) + a3_s*np.sin(3.0*omega_s*time)
#     return s0t

# def uu0_known(time):
#     omega_u = 2*np.pi/(58*365*60*60*24)
#     a0_u=775.302217
#     a1_u= -60.638295
#     a2_u= -139.314257
#     a3_u=-16.080739
#     u0t = a0_u + a1_u*np.sin(1.0*omega_u*time) + a2_u*np.sin(2.0*omega_u*time) + a3_u*np.sin(3.0*omega_u*time)
#     return u0t 
# uu0_const = 775.302217