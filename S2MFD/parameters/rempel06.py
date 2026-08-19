"""Rempel (2006) 参照モデルのパラメタ.

Rempel, M. 2006, ApJ, 647, 662
  "Flux-Transport Dynamos with Lorentz Force Feedback on Differential Rotation
   and Meridional Flow: Saturation Mechanism and Torsional Oscillations"

背景成層・Λ効果・超断熱度は Rempel (2005, ApJ 622, 1320) の case 1 に従う。
Rempel 2006 §2.1 が case 1 から変更しているのは n=3, Λ0=1,
nu_t = kappa_t = 3e8 m^2/s の 3 点のみ。

式の出典と転記は ``doc/dev_records/2026-08-19_rempel_equations.md`` を参照。
数値実装は齋藤 (2024, 名古屋大学修士論文) の保存形離散化を踏襲する。

単位は S2MFD 本体と同じ CGS (論文は SI なので換算済み)。
"""
import numpy as np

from S2MFD.parameters.defaults import *  # noqa: F401,F403

# --- 実行モード -----------------------------------------------------------
dynamics = 'dynamic'

# --- 計算領域 -------------------------------------------------------------
# Rempel 2005 §2.6 は r = 0.65 - 0.985 RSUN、動径 108 x 緯度 72 (北半球のみ)。
# S2MFD は全球 [0, pi] を解くので緯度方向は 2 倍取る。
#
# 上部境界を 0.96 RSUN にしている理由
# ------------------------------------
# Rempel の 0.985 RSUN では、SSP-RK2 + 中心差分の本実装は不安定になる。
# Rempel(2005) のポリトロープは 1.0007 RSUN で密度ゼロに達するので、
# 0.985 RSUN では xi = 0.039 まで落ち、1 セルあたりの密度変化が大きい。
# 実測: 0.985 RSUN では nx=64 でも nx=128 (dr/Hp=0.35) でも発散し、
# 0.96 RSUN なら nx=64 で安定する (doc/dev_records の作業記録を参照)。
# Rempel は MacCormack の交互風上/風下差分に内在する強い数値散逸を
# 持つため 0.985 RSUN まで解ける。齋藤 (2024) が 0.96 RSUN を採用して
# いるのも同じ事情と考えられる。
#
# 0.985 RSUN を再現したい場合は、上部で格子を伸縮させるか、上部境界近傍で
# 散逸を強める必要がある (未実装)。
rrmin = 0.65*RSUN
rrmax = 0.96*RSUN
rrmax_rempel = 0.985*RSUN   # 原論文の値 (参考)
thmin = 0.0
thmax = np.pi

ix = 108
jx = 144

# --- 熱力学 ---------------------------------------------------------------
gamma = 5.0/3.0

# 対流層底 (0.71 RSUN) での参照値 [CGS]。Rempel 2005 式 (21)-(24)。
rr_bc = 0.71*RSUN
ro_bc = 0.2       # 200 kg/m^3
pr_bc = 6.0e13    # 6e12 Pa
gr_bc = 5.2e4     # 520 m/s^2
tm_bc = 1.82e6    # [K] (論文に明示がないので齋藤 2024 の値)

# --- 超断熱度 delta(r) (Rempel 2005 式 25-26, case 1) --------------------
# delta = grad - grad_ad。正が超断熱 (対流不安定)、負が亜断熱。
# case 1 は対流層が断熱 (delta_conv = 0) でオーバーシュート層のみ亜断熱。
delta_os = -1.5e-5       # オーバーシュート層の値
delta_cz = 0.0           # 対流層バルク (case 1 は 0)
delta_top = 0.0          # 上端 (case 1 は 0)
r_tran = 0.725*RSUN      # オーバーシュート層への遷移半径
d_tran = 0.0125*RSUN
d_top = 0.0125*RSUN
r_sub = 0.8*RSUN         # delta_cz != 0 のときだけ意味を持つ

# --- 音速抑制法 (RSST) ----------------------------------------------------
# Rempel 2005 式 (35) の規約: 連続の式に 1/zeta^2 が掛かる。zeta = 100。
# (Rempel 2006 式 12 は逆数規約 zeta = 0.01 を使う。混同しないこと。)
rsst_type = 'const'
rsst_zeta = 100.0

# --- 乱流粘性・熱伝導 (Rempel 2005 式 27-30) ------------------------------
# Rempel 2006 参照モデル: nu0 = kappa0 = 3e8 m^2/s
nu0 = 3.0e12       # [cm^2/s]
kappa0 = 3.0e12    # [cm^2/s]
d_kn = 0.025*RSUN  # d_{kappa,nu}
alpha_kn = 0.1     # r = r_tran での拡散係数の比
d_bc = 0.0125*RSUN

# 放射層でのフロア。Rempel 2005 §2.4:
# 「Lambda 効果による角運動量輸送には式 (27) のプロファイルを使うが、
#   レイノルズ応力の拡散項に使う粘性は対流層値の 2%、熱伝導は 0.2% にする」
# Omega1 = 0 の下部境界との間にせん断層 (タコクライン) を現実的な時間で
# 形成させるための措置。
nu_floor_frac = 0.02
kappa_floor_frac = 0.002

# 数値安定性のための増幅係数 (論文からの逸脱を明示するための係数)。
#
# 当初、論文どおりの値では 0.4 年程度で発散したため kappa を 3 倍にしていたが、
# 原因は散逸加熱を「Omega * dq」で局所評価していたことだった。正しい局所形
# 「-F . grad(Omega)」に直したところ、論文値 (1.0) のまま安定になった。
# 詳細は doc/dev_records/2026-08-20_rempel2006_worklog.md §4.4。
nu_numerical_factor = 1.0
kappa_numerical_factor = 1.0

# --- 角運動量輸送 (Lambda 効果, Rempel 2005 式 31-33) ---------------------
# Rempel 2006 参照モデル: n = 3, Lambda0 = 1, lambda = 15 度
lambda0 = 1.0
lambda_n = 3.0            # lambda_tilt > 0 のときは n > 2 が必要 (極での正則性)
lambda_d = 0.025*RSUN
lambda_tilt_deg = 15.0

# --- 回転 -----------------------------------------------------------------
om0 = 413.0e-9*2.0*np.pi   # [rad/s]

# --- 磁気拡散 (Rempel 2006 式 13-15) --------------------------------------
diffusive_type = 'R06'
eta_c = 1.0e9      # 1e5 m^2/s  (放射層)
eta_bc = 1.0e10    # 1e6 m^2/s  (対流層底)
eta_cz = 1.0e12    # 1e8 m^2/s  (対流層)
r_cz = 0.875*RSUN
d_cz = 0.05*RSUN

# --- Babcock-Leighton alpha 効果 (Rempel 2006 式 16-19) -------------------
alpha_type = 'R06'
alpha0 = 12.5             # 0.125 m/s [cm/s]
d_alpha = 0.05*RSUN
# B_phi を平均する放物線カーネル h(r): 0.71 - 0.76 RSUN でゼロ、0.735 でピーク
r_h_bot = 0.71*RSUN
r_h_top = 0.76*RSUN

# --- 境界条件 -------------------------------------------------------------
# 'uniform_rotation' : Rempel 2005/2006。下部境界で Omega1 = 0 を課して
#                      タコクラインを強制する。系は角運動量について閉じない
#                      ので、境界フラックスを積算して収支を検証する。
# 'stress_free'      : 齋藤 (2024)。閉じた系で ∫q_L が machine precision 保存。
angmom_bottom_bc = 'uniform_rotation'

# --- 人工拡散 -------------------------------------------------------------
# Rempel は MacCormack の交互風上/風下差分に内在する数値散逸を使っており、
# 人工粘性を明示的には入れていない。S2MFD は SSP-RK2 + 中心差分で数値散逸を
# 持たないため、Hotta (2017) の slope-limited diffusion を保存形で入れる。
# 散逸した運動エネルギー・磁場エネルギーはエントロピー方程式に戻す。
artificial_diffusion = True
sld_coefficient = 1.0
# 部分的な風上化の下限。Rempel の MacCormack が持つ暗黙の数値散逸に相当する量を
# 明示的に入れるための係数。0 なら純粋な SLD、1 なら完全な Rusanov (1次風上)。
#
# 値の選び方: この係数は実効的な拡散係数 (1/2)*floor*c_eff*dx を生む。
# 64x64 では floor=0.1 で 4.3e12 となり物理粘性 nu_t=3e12 を超えてしまい、
# 差動回転を有意に削る。実測 (64x64, 6年):
#     floor=0.1  -> DR=+0.0244 (人工拡散 4.3e12)
#     floor=0.03 -> DR=+0.0344 (人工拡散 1.3e12)
#     floor=0.01 -> DR=+0.0366 (人工拡散 4.3e11)
# 0.01 で頭打ちになるので既定値とする。
sld_floor = 0.01
# 特性速度 c = |v| + sld_cs_factor * c_s,eff + v_A の、抑制音速に掛ける係数。
# 0.1-0.3 が標準。抑制後とはいえ音速は流れより 2 桁速いので、そのまま使うと
# 人工拡散がモデルの依存する低拡散領域 (オーバーシュート層の kappa_t、
# 放射層の nu_dif) を潰してしまう。
sld_cs_factor = 0.2

# --- CFL ------------------------------------------------------------------
cfl_safety = 0.2

# --- 時間 -----------------------------------------------------------------
tend = 30*365*d2s
dtout = 10*d2s
