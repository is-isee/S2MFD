from S2MFD.parameters.defaults import *

tend = 91000*d2s
# boundary condition
boundary_condition_type = 'potential'
datadir = 'data_fousamp_s0/'


# # 正解
# # 35付近
def so0_time_dependent(time, ett, RSUN):
    factor = 1.0
    a0 = 35
    a1 = 1  *factor
    a2 = 3  *factor
    a3 = 0.5*factor
    b1 = 3  *factor
    b2 = 1.2*factor
    b3 = 4  *factor
    
    omega = 2*np.pi/(21*365*60*60*1000)  # 1回転にかかる時間(秒) = 21年
    """時間 t に応じて so0 を変化させる関数(フーリエ級数)"""
    s0t = (a0   + a1*np.cos(1.0*omega*time) + b1*np.sin(1.0*omega*time)\
                + a2*np.cos(2.0*omega*time) + b2*np.sin(2.0*omega*time)\
                + a3*np.cos(3.0*omega*time) + b3*np.sin(3.0*omega*time)) * ett / RSUN
    return s0t
