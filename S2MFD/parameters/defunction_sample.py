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

# def uu0_time_dependent(time, ett, RSUN):
#     rey = 700  # 基本のスケール
#     # return rey * ett / RSUN * (1 + 0.5*np.sin(2 * np.pi * time / (5e9)))  # 例: コサイン波で変化
#     return rey * ett / RSUN

# def so0_time_dependent(A,omega,B,time):
#     """時間 t に応じて so0 を変化させる関数"""
#     return A * ett / RSUN * (np.sin(2 * np.pi * time / (omega))) + B * ett / RSUN

def so0_time_dependent(A,omega,B,C,time):
    """時間 t に応じて so0 を変化させる関数"""
    return (A * (np.sin((2 * np.pi * omega * time)-C)) + B) * ett / RSUN


# def uu0_time_dependent(A,omega,B,time):
#     """時間 t に応じて uu0 を変化させる関数"""
#     return (A * (np.sin((2 * np.pi * omega * time)-C)) + B) * ett / RSUN

# 時間依存の so0 と uu0 を計算
# if 'simulation' not in globals():
#     simulation = None
    
# if simulation == None:
#     simulation = type('Simulation', (object,), {})()  # ダミーのオブジェクトを作成
#     simulation.time = 0  # 初期時間を設定
# so0 = so0_time_dependent(simulation.time, ett, RSUN)
# uu0 = uu0_time_dependent(simulation.time, ett, RSUN)