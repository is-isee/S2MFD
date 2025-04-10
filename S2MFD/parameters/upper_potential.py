from S2MFD.parameters.defaults import *

boundary_condition_type = 'potential'

def so0_time_dependent(time, ett, RSUN):
    """時間 t に応じて so0 を変化させる関数"""
    cso = 35  # 基本のスケール
    # return cso * ett / RSUN * (1 + np.sin(2 * np.pi * time / (1e10)))  # 例: サイン波で変化
    return cso * ett / RSUN

def uu0_time_dependent(time, ett, RSUN):
    """時間 t に応じて uu0 を変化させる関数"""
    rey = 700  # 基本のスケール
    return rey * ett / RSUN
