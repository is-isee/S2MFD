from S2MFD.parameters.defaults import *

tend = 91000*d2s
# boundary condition
boundary_condition_type = 'potential'
rrmin = 0.60*RSUN



# # 正解
# # 35付近
# def so0_time_dependent(time, ett, RSUN):
#     factor = 1.0
#     a0 = 35
#     a1 = 1  *factor
#     a2 = 3  *factor
#     a3 = 0.5*factor
#     b1 = 3  *factor
#     b2 = 1.2*factor
#     b3 = 4  *factor
    
#     omega = 1/(693782000*4.3)
#     """時間 t に応じて so0 を変化させる関数(フーリエ級数)"""
#     s0t = (a0   + a1*np.cos(2*np.pi*omega*time) + b1*np.sin(2*np.pi*omega*time)\
#                 + a2*np.cos(4*np.pi*omega*time) + b2*np.sin(4*np.pi*omega*time)\
#                 + a3*np.cos(6*np.pi*omega*time) + b3*np.sin(6*np.pi*omega*time)) * ett / RSUN
#     return s0t


# # 正解
# # 700付近
# def uu0_time_dependent(time, ett, RSUN):
#     factor = 1.0
#     a0 = 600
#     a1 = 140*factor
#     a2 = 21 *factor
#     a3 = 70 *factor
#     b1 = 44 *factor
#     b2 = 90 *factor
#     b3 = 20 *factor
#     omega = 1/(693782000*6)  # 1回転にかかる時間(秒) = 693782000秒
#     """時間 t に応じて so0 を変化させる関数(フーリエ級数)"""
#     u0t = (a0   + a1*np.cos(2*np.pi*omega*time) + b1*np.sin(2*np.pi*omega*time)\
#                 + a2*np.cos(4*np.pi*omega*time) + b2*np.sin(4*np.pi*omega*time)\
#                 + a3*np.cos(6*np.pi*omega*time) + b3*np.sin(6*np.pi*omega*time)) * ett / RSUN
#     return u0t


def so0_time_dependent(a0_s,a1_s,a2_s,a3_s,b1_s,b2_s,b3_s,omega_s,time):
    """時間 t に応じて so0 を変化させる関数(フーリエ級数)"""
    s0t = (a0_s   + a1_s*np.cos(2*np.pi*omega_s*time) + b1_s*np.sin(2*np.pi*omega_s*time)\
                  + a2_s*np.cos(4*np.pi*omega_s*time) + b2_s*np.sin(4*np.pi*omega_s*time)\
                  + a3_s*np.cos(6*np.pi*omega_s*time) + b3_s*np.sin(6*np.pi*omega_s*time)) * ett / RSUN
    return s0t

def uu0_time_dependent(a0_u,a1_u,a2_u,a3_u,b1_u,b2_u,b3_u,omega_u,time):
    """時間 t に応じて uu0 を変化させる関数(フーリエ級数)"""
    u0t = (a0_u   + a1_u*np.cos(2*np.pi*omega_u*time) + b1_u*np.sin(2*np.pi*omega_u*time)\
                  + a2_u*np.cos(4*np.pi*omega_u*time) + b2_u*np.sin(4*np.pi*omega_u*time)\
                  + a3_u*np.cos(6*np.pi*omega_u*time) + b3_u*np.sin(6*np.pi*omega_u*time)) * ett / RSUN
    return u0t
