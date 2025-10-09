from S2MFD.parameters.defaults import *

tend = 91000*d2s
# boundary condition
boundary_condition_type = 'potential'

def uu0_time_dependent(as_u,ae_u,a1_u,a2_u,a4_u,time):
    # 最適化区間の2倍で定義
    inf_year = 52
    omega_u = 2*np.pi/(inf_year*365*60*60*24*2)
    # 線形関数
    lin  = (as_u*(time[-1]-time)+ae_u*(time-time[0]))/(time[-1]-time[0])
    u0t = lin + a1_u*np.sin(1.0*omega_u*time) + a2_u*np.sin(2.0*omega_u*time) + a4_u*np.sin(4.0*omega_u*time)
    return u0t
# TODO s0の推定関数を書くこと
# TODO uu0_knownを書き直すこと

# def uu0_known(time):
#     omega_u = 2*np.pi/(52*365*60*60*24)
#     a0_u=791.013294
#     a1_u=73.513338
#     a2_u=-138.214256
#     a3_u=33.452918
#     u0t = a0_u + a1_u*np.sin(1.0*omega_u*time) + a2_u*np.sin(2.0*omega_u*time) + a3_u*np.sin(3.0*omega_u*time)
#     return u0t 
# uu0_const = 791.013294