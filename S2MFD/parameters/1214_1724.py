from S2MFD.parameters.defaults import *

tend = 91000*d2s
# boundary condition
boundary_condition_type = 'potential'

def uu0_time_dependent(as_u,ae_u,a1_u,a2_u,a4_u,time):
    # 最適化区間の2倍で定義
    inf_year = 55.890410958902976
    omega_u = 2*np.pi/(inf_year*365*60*60*24*2)
    T_s = 0.0
    T_e = inf_year*365*60*60*24
    # 線形関数
    lin  = (as_u*(T_e-time)+ae_u*(time-T_s))/(T_e-T_s)
    u0t = lin + a1_u*np.sin(1.0*omega_u*time) + a2_u*np.sin(2.0*omega_u*time) + a4_u*np.sin(4.0*omega_u*time)
    return u0t

# def so0_time_dependent(as_s,ae_s,a1_s,a2_s,a4_s,time):
#     # 最適化区間の2倍で定義
#     inf_year = 55.890410958902976
#     omega_s = 2*np.pi/(inf_year*365*60*60*24*2)
#     T_s = 0.0
#     T_e = inf_year*365*60*60*24
#     # 線形関数
#     lin  = (as_s*(T_e-time)+ae_s*(time-T_s))/(T_e-T_s)
#     so0t = lin + a1_s*np.sin(1.0*omega_s*time) + a2_s*np.sin(2.0*omega_s*time) + a4_s*np.sin(4.0*omega_s*time)
#     return so0t

# def uu0_known(time):
#     inf_year = 55.890410958902976
#     omega_u = 2*np.pi/(inf_year*365*60*60*24*2)
#     as_u= 753.360841
#     ae_u= 798.717979
#     a1_u= -45.854816
#     a2_u= 2.256662
#     a4_u= -38.634266
#     T_s = 0.0
#     T_e = inf_year*365*60*60*24
#     lin  = (as_u*(T_e-time)+ae_u*(time-T_s))/(T_e-T_s)
#     u0t = lin + a1_u*np.sin(1.0*omega_u*time) + a2_u*np.sin(2.0*omega_u*time) + a4_u*np.sin(4.0*omega_u*time)

#     return u0t 
# uu0_const = 753.360841
