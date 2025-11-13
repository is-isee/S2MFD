from S2MFD.parameters.defaults import *

tend = (90640.45+20075)*d2s
# boundary condition
boundary_condition_type = 'potential'
datadir = 'num_results/data_ground_3/'

# # 正解
# # 700付近
def uu0_time_dependent(time, ett, RSUN):
    # 最適化区間の2倍で定義
    time = time - 7.83138412e+09
    inf_year = 55
    omega_u = 2*np.pi/(inf_year*365*60*60*24*2)
    T_s = 0.0
    T_e = inf_year*365*60*60*24
    as_u=694.88761297
    ae_u=550
    a1_u=-96
    a2_u=38
    a4_u=22
    # 線形関数
    lin  = (as_u*(T_e-time)+ae_u*(time-T_s))/(T_e-T_s)
    u0t = lin + a1_u*np.sin(1.0*omega_u*time) + a2_u*np.sin(2.0*omega_u*time) + a4_u*np.sin(4.0*omega_u*time)

    return u0t

def so0_time_dependent(time, ett, RSUN):
    # 最適化区間の2倍で定義
    time = time - 7.83138412e+09
    inf_year = 55
    omega_u = 2*np.pi/(inf_year*365*60*60*24*2)
    T_s = 0.0
    T_e = inf_year*365*60*60*24
    as_s=50.28735632
    ae_s=58
    a1_s=15
    a2_s=3.35
    a4_s=1.2
    # 線形関数
    lin  = (as_s*(T_e-time)+ae_s*(time-T_s))/(T_e-T_s)
    s0t = lin + a1_s*np.sin(1.0*omega_u*time) + a2_s*np.sin(2.0*omega_u*time) + a4_s*np.sin(4.0*omega_u*time)

    return s0t