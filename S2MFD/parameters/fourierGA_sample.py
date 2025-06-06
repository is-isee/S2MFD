from S2MFD.parameters.defaults import *

tend = 91000*d2s
# boundary condition
boundary_condition_type = 'potential'
rrmin = 0.60*RSUN



# 正解
def so0_time_dependent(time, ett, RSUN):
    a0 = 0.0
    a1 = 0.0
    a2 = 0.0
    a3 = 0.0
    b1 = 0.0
    b2 = 0.0
    b3 = 0.0
    omega = 1/693782000  # 1回転にかかる時間(秒) = 693782000秒
    """時間 t に応じて so0 を変化させる関数(フーリエ級数)"""
    s0t = (a0*0.5   + a1*cos(2*pi*omega*time) + b1*sin(2*pi*omega*time)\
                    + a2*cos(4*pi*omega*time) + b2*sin(4*pi*omega*time)\
                    + a3*cos(6*pi*omega*time) + b3*sin(6*pi*omega*time)) * ett / RSUN
    return s0t


# 正解
def uu0_time_dependent(time, ett, RSUN):
    a0 = 0.0
    a1 = 0.0
    a2 = 0.0
    a3 = 0.0
    b1 = 0.0
    b2 = 0.0
    b3 = 0.0
    omega = 1/693782000  # 1回転にかかる時間(秒) = 693782000秒
    """時間 t に応じて so0 を変化させる関数(フーリエ級数)"""
    u0t = (a0*0.5   + a1*cos(2*pi*omega*time) + b1*sin(2*pi*omega*time)\
                    + a2*cos(4*pi*omega*time) + b2*sin(4*pi*omega*time)\
                    + a3*cos(6*pi*omega*time) + b3*sin(6*pi*omega*time)) * ett / RSUN
    return u0t


# def so0_time_dependent(a0,a1,b1,a2,b2,a3,b3,omega,time):
#     """時間 t に応じて so0 を変化させる関数(フーリエ級数)"""
#     s0t = (a0*0.5   + a1*cos(2*pi*omega*time) + b1*sin(2*pi*omega*time)\
#                     + a2*cos(4*pi*omega*time) + b2*sin(4*pi*omega*time)\
#                     + a3*cos(6*pi*omega*time) + b3*sin(6*pi*omega*time)) * ett / RSUN
#     return s0t

# def uu0_time_dependent(a0,a1,b1,a2,b2,a3,b3,omega,time):
#     """時間 t に応じて uu0 を変化させる関数(フーリエ級数)"""
#     u0t = (a0*0.5   + a1*cos(2*pi*omega*time) + b1*sin(2*pi*omega*time)\
#                     + a2*cos(4*pi*omega*time) + b2*sin(4*pi*omega*time)\
#                     + a3*cos(6*pi*omega*time) + b3*sin(6*pi*omega*time)) * ett / RSUN
#     return u0t
