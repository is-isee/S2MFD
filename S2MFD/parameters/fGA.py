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
    a1 = 27.99097933654422  *factor
    a2 = 1.9834880111338207 *factor
    a3 = 0.5*factor
    b1 = 3  *factor
    b2 = 1.2*factor
    b3 = 4  *factor
    
    a0 = 27.99097933654422
    a1 = 1.9834880111338207
    a2 = 5.682653986905811
    a3 = 0.28556955709611864
    b1 = 3.4980089430453107
    b2 = 4.87404229158737
    b3 = 2.783061193095837
    omega = 5.933729599592805e-10,
    
    omega = 1/(693782000*4.3)
    """時間 t に応じて so0 を変化させる関数(フーリエ級数)"""
    s0t = (a0   + a1*np.cos(2*np.pi*omega*time) + b1*np.sin(2*np.pi*omega*time)\
                + a2*np.cos(4*np.pi*omega*time) + b2*np.sin(4*np.pi*omega*time)\
                + a3*np.cos(6*np.pi*omega*time) + b3*np.sin(6*np.pi*omega*time)) * ett / RSUN
    return s0t


# 正解
# 700付近
def uu0_time_dependent(time, ett, RSUN):
    factor = 1.0
    a0 = 600
    a1 = 140*factor
    a2 = 21 *factor
    a3 = 70 *factor
    b1 = 44 *factor
    b2 = 90 *factor
    b3 = 20 *factor
    omega = 1/(693782000*6)  # 1回転にかかる時間(秒) = 693782000秒
    
    a0 = 1049.6602290996639
    a1 = 283.82284088293864
    a2 = 19.621336069564723
    a3 = 175.26383537551823
    b1 = 151.74341534275644
    b2 = 40.00927199241416
    b3 = 143.23084385429942
    omega = 1.4999005622304982e-10

    """時間 t に応じて so0 を変化させる関数(フーリエ級数)"""
    u0t = (a0   + a1*np.cos(2*np.pi*omega*time) + b1*np.sin(2*np.pi*omega*time)\
                + a2*np.cos(4*np.pi*omega*time) + b2*np.sin(4*np.pi*omega*time)\
                + a3*np.cos(6*np.pi*omega*time) + b3*np.sin(6*np.pi*omega*time)) * ett / RSUN
    return u0t

