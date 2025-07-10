from S2MFD.parameters.defaults import *

tend = 91000*d2s
# boundary condition
boundary_condition_type = 'potential'
datadir = 'data_fousamp_u0/'

# # 正解
# # 700付近
def uu0_time_dependent(time, ett, RSUN):
    factor = 1.0
    a0 = 600
    a1 = 140*factor
    a2 = 21 *factor
    a3 = 70 *factor
    b1 = 44 *factor
    b2 = 90 *factor
    b3 = 20 *factor
    omega = 2*np.pi/(18*365*60*60)  # 多分うまくいかない
    # omega = 2*np.pi/(18*365*60*60*1000)  # こっちを採用予定
    """時間 t に応じて so0 を変化させる関数(フーリエ級数)"""
    u0t = (a0   + a1*np.cos(1.0*omega*time) + b1*np.sin(1.0*omega*time)\
                + a2*np.cos(2.0*omega*time) + b2*np.sin(2.0*omega*time)\
                + a3*np.cos(3.0*omega*time) + b3*np.sin(3.0*omega*time)) * ett / RSUN
    return u0t


