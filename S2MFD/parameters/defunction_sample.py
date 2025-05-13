from S2MFD.parameters.defaults import *

tend = 91000*d2s
# boundary condition
boundary_condition_type = 'potential'
rrmin = 0.60*RSUN


"""
正解
def so0_time_dependent(time, ett, RSUN):
    A = 10  # 基本のスケール
    omega = 693792000  # 周期
    B = 35  # 基本のスケール
    return A * ett / RSUN * (np.sin(2 * np.pi * time / (omega))) + B * ett / RSUN # 例: サイン波で変化
    # return cso * ett / RSUN
"""

"""
# 正解
def uu0_time_dependent(time, ett, RSUN):
    A = 70
    B = 700
    C = np.pi/6
    omega = 693792000
    return A * ett / RSUN * (np.sin(2 * np.pi * time / (omega))) + B * ett / RSUN # 例: サイン波で変化
    # return rey * ett / RSUN
"""

# def so0_time_dependent(A,omega,B,C,time):
#     """時間 t に応じて so0 を変化させる関数"""
#     return (A * (np.sin((2 * np.pi * omega * time)-C)) + B) * ett / RSUN

def uu0_time_dependent(A,omega,B,C,time):
    """時間 t に応じて uu0 を変化させる関数"""
    return A * ett / RSUN * (np.sin(2 * np.pi * time * (omega))) + B * ett / RSUN
