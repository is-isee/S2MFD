from S2MFD.parameters.defaults import *

tend = 91000*d2s
# boundary condition
boundary_condition_type = 'potential'
datadir = 'data_sinsamp_u0/'

# # 正解
# # 700付近
def uu0_time_dependent(time, ett, RSUN):
    a0 = 700
    a1 = 70 
    a2 = 65
    a3 = -90

    # omega = 2*np.pi/(18*365*60*60)  # 多分うまくいかない
    omega = 2*np.pi/(150*365*60*60*24)  # こっちを採用予定
    """時間 t に応じて so0 を変化させる関数(フーリエ級数)"""
    u0t = a0 + a1*np.sin(1.0*omega*time) + a2*np.sin(2.0*omega*time) + a3*np.sin(3.0*omega*time)
    return u0t
