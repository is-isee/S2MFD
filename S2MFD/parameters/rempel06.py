"""Rempel (2006) の平均場ダイナモ + 差動回転モデルのパラメタ.

Rempel, M. 2006, ApJ, 647, 662
  "Flux-Transport Dynamos with Lorentz Force Feedback on Differential
   Rotation and Meridional Flow: Nonlinear Evolution and Saturation"

背景成層と対流層底での参照値は Rempel (2005, ApJ, 622, 1320) に従う。
数値実装は齋藤 (2024, 名古屋大学修士論文) の平均場モデルコードを踏襲し、
角運動量と質量を machine precision で保存する保存形離散化を用いる。

単位系は S2MFD 本体と同じ CGS。
"""
import numpy as np

from S2MFD.parameters.defaults import *  # noqa: F401,F403

# --- 実行モード -----------------------------------------------------------
dynamics = 'dynamic'

# --- 計算領域 -------------------------------------------------------------
# 上部境界は 0.96 RSUN。Rempel(2005) のポリトロープは 1.003 RSUN で密度が
# ゼロになるため、1.0 RSUN までは取れない (Stratification が検出して弾く)。
rrmin = 0.65*RSUN
rrmax = 0.96*RSUN
thmin = 0.0
thmax = np.pi

ix = 128
jx = 128

# --- 熱力学 ---------------------------------------------------------------
gamma = 5.0/3.0

# 対流層底 (0.71 RSUN) での参照値 [CGS]
rr_bc = 0.71*RSUN
ro_bc = 0.2       # [g/cm^3]
pr_bc = 6.0e13    # [dyn/cm^2]
tm_bc = 1.82e6    # [K]
gr_bc = 5.2e4     # [cm/s^2]

# 超断熱度 delta = grad - grad_ad。対流層では正の微小量。
delta_superadiabatic = 1.0e-6

# --- 音速抑制法 (RSST) ----------------------------------------------------
# 実効音速は cs/rsst_zeta。齋藤 (2024) は一定値 100 を用いる。
#   'const'      : 一定値 (既定)
#   'uniform_dt' : 実効音速を格子幅に比例させて dt を動径方向に一様化する
rsst_type = 'const'
rsst_zeta = 100.0

# --- 粘性 -----------------------------------------------------------------
# 乱流粘性係数 [cm^2/s]。Rempel (2006) は磁気拡散と同程度の値を用いる。
nu_turb = 5.0e12

# --- 角運動量輸送 (Lambda 効果) -------------------------------------------
# Rempel (2005) 式 (14)-(16) 相当。差動回転を維持する非等方レイノルズ応力。
lambda0 = 1.0
lambda_r_type = 'R05'
lambda_th_type = 'R05'

# --- 回転 -----------------------------------------------------------------
om0 = 413.0e-9*2.0*np.pi   # 剛体回転部分 [rad/s]

# --- CFL ------------------------------------------------------------------
# 音速抑制後も音速が CFL を決めるため、運動学的ダイナモより dt はずっと小さい。
cfl_safety = 0.2

# --- 時間 -----------------------------------------------------------------
tend = 30*365*d2s      # 30 年
dtout = 10*d2s
