from S2MFD.parameters.defaults import *

tend = 91000*d2s
# boundary condition
boundary_condition_type = 'potential'
rrmin = 0.60*RSUN



# 正解
# 35付近
def so0_time_dependent(time, ett, RSUN):
    factor = 1.0
    a0 = 35
    a1 = 1*factor
    a2 = 3*factor
    a3 = 0.5*factor
    b1 = 3*factor
    b2 = 1.2*factor
    b3 = 4*factor
    
    omega = 1/(693782000*4.3)
    """時間 t に応じて so0 を変化させる関数(フーリエ級数)"""
    s0t = (a0   + a1*cos(2*pi*omega*time) + b1*sin(2*pi*omega*time)\
                + a2*cos(4*pi*omega*time) + b2*sin(4*pi*omega*time)\
                + a3*cos(6*pi*omega*time) + b3*sin(6*pi*omega*time)) * ett / RSUN
    return s0t


# 正解
# 700付近
def uu0_time_dependent(time, ett, RSUN):
    factor = 1.0
    a0 = 600
    a1 = 140*factor
    a2 = 21*factor
    a3 = 70*factor
    b1 = 44*factor
    b2 = 90*factor
    b3 = 20*factor
    omega = 1/(693782000*6)  # 1回転にかかる時間(秒) = 693782000秒
    """時間 t に応じて so0 を変化させる関数(フーリエ級数)"""
    u0t = (a0   + a1*cos(2*pi*omega*time) + b1*sin(2*pi*omega*time)\
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
