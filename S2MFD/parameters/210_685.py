from S2MFD.parameters.defaults import *

tend = 91000*d2s
# boundary condition
boundary_condition_type = 'potential'

def uu0_time_dependent(as_u,ae_u,a1_u,a2_u,a4_u,time):
    # 最適化区間の2倍で定義
    inf_year = 52.05479452054692
    omega_u = 2*np.pi/(inf_year*365*60*60*24*2)
    T_s = 0.0
    T_e = inf_year*365*60*60*24
    # 線形関数
    lin  = (as_u*(T_e-time)+ae_u*(time-T_s))/(T_e-T_s)
    u0t = lin + a1_u*np.sin(1.0*omega_u*time) + a2_u*np.sin(2.0*omega_u*time) + a4_u*np.sin(4.0*omega_u*time)
    return u0t
# TODO s0の推定関数を書くこと
# TODO uu0_knownを書き直すこと

# def uu0_known(time):
#     omega_u = 2*np.pi/(52*365*60*60*24)
#     a0_u=724.027707 
#     a1_u=21.742294 
#     a2_u=26.276269
#     a3_u=-32.225917
#     u0t = a0_u + a1_u*np.sin(1.0*omega_u*time) + a2_u*np.sin(2.0*omega_u*time) + a3_u*np.sin(3.0*omega_u*time)
#     return u0t 
# uu0_const = 724.027707 