from S2MFD.parameters.defaults import *

tend = 91000*d2s
# boundary condition
boundary_condition_type = 'potential'
datadir = 'data_sinsamp_u0s0/'

# # 正解
# # 700付近
def uu0_time_dependent(time, ett, RSUN):
    a0 = 750
    a1 = 35 
    a2 = 32
    a3 = -45

    # omega = 2*np.pi/(18*365*60*60)  # 多分うまくいかない
    omega = 2*np.pi/(150*365*60*60*24)  # こっちを採用予定
    """時間 t に応じて so0 を変化させる関数(フーリエ級数)"""
    u0t = a0 + a1*np.sin(1.0*omega*time) + a2*np.sin(2.0*omega*time) + a3*np.sin(3.0*omega*time)
    return u0t

def so0_time_dependent(time, ett, RSUN):
    a0 = 60.847479
    a1 = 6.707831
    a2 = -7.504918
    a3 = -4.223816
    
    omega = 2*np.pi/(100*365*60*60*24)  # 1回転にかかる時間(秒) = 21年　（こっちが正しいが、実際お手本データとして使っているのは上）
    """時間 t に応じて so0 を変化させる関数(フーリエ級数)"""
    s0t = a0 + a1*np.sin(1.0*omega*time) + a2*np.sin(2.0*omega*time) + a3*np.sin(3.0*omega*time)
    return s0t