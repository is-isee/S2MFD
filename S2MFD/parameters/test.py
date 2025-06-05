from S2MFD.parameters.defaults import *

tend = 91000*d2s
# boundary condition
boundary_condition_type = 'potential'
rrmin = 0.60*RSUN


# 正解
def so0_time_dependent(time, ett, RSUN):
    A = 17.596297853488114  # 基本のスケール
    omega = 1.431279698269089e-09  # 周期
    B = 32.72946658405157  # 基本のスケール
    C = 5.612100922032009
    return (A * (np.sin((2 * np.pi * omega * time)-C)) + B) * ett / RSUN # 例: サイン波で変化
    # return cso * ett / RSUN


def uu0_time_dependent(time, ett, RSUN):
    A = 74.98520249288524
    B = 693.2501478794433
    C = 7.280718810374227
    omega = 1.469036272389512e-09
    return (A * (np.sin((2 * np.pi * omega * time)-C)) + B) * ett / RSUN # 例: サイン波で変化
    # return rey * ett / RSUN

