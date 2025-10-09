from S2MFD.parameters.defaults import *

tend = 91000*d2s
# boundary condition
boundary_condition_type = 'potential'

def uu0_time_dependent(as_u,ae_u,a1_u,a2_u,a4_u,time):
    # 最適化区間の2倍で定義
    inf_year = 55
    omega_u = 2*np.pi/(inf_year*365*60*60*24*2)
    # 線形関数
    lin  = (as_u*(time[-1]-time)+ae_u*(time-time[0]))/(time[-1]-time[0])
    u0t = lin + a1_u*np.sin(1.0*omega_u*time) + a2_u*np.sin(2.0*omega_u*time) + a4_u*np.sin(4.0*omega_u*time)
    return u0t
# TODO s0の推定関数を書くこと
# TODO uu0_knownを書き直すこと

# def uu0_known(time):
#     omega_u = 2*np.pi/(55*365*60*60*24)
#     a0_u=678.090293
#     a1_u=-9.409831
#     a2_u=57.470681
#     a3_u=-27.533206
#     u0t = a0_u + a1_u*np.sin(1.0*omega_u*time) + a2_u*np.sin(2.0*omega_u*time) + a3_u*np.sin(3.0*omega_u*time)
#     return u0t 
# uu0_const = 678.090293