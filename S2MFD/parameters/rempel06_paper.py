"""Rempel (2006) 参照モデルを**論文どおりの設定**で解くパラメタファイル。

``rempel06.py`` との違いは計算領域と格子だけで、物理パラメタは同じ。
``rempel06.py`` は数値安定性のために上端を 0.96 RSUN に切り詰め、全球
[0, pi] を解く既定になっている。こちらは論文に合わせる:

Rempel (2005) §2.6
    "The computational domain extends in latitude from equator to pole and
     in radius from r = 0.65 to 0.985 R_sun. ... a moderate resolution of
     around 108 grid points in radius and 72 grid points in latitude is
     sufficient."

Rempel (2006) §2.2
    "We restrict our simulations to one hemisphere and impose the dipole
     symmetry through our equatorial boundary condition."

なぜ領域を論文どおりにする必要があるか
--------------------------------------
Lambda 効果 (2005 式 33) の ``tanh((r_max - r)/d)`` と alpha 効果
(2006 式 17) の ``max[0, 1 - (r - r_max)^2/d_alpha^2]`` は、どちらも
r_max = 0.985 RSUN を基準にしている。領域を 0.96 に切り詰めると
**駆動の位置まで 0.025 RSUN 内側に動く** (論文 §2.2 の
"confines the poloidal source term above r = 0.935 R_sun" が 0.91 になる)。

0.985 RSUN は密度が薄くなって SSP-RK2 + 中心差分では不安定になりやすいので、
上端に格子を集中させた 2 点集中格子を使う (r985s / r985t で 25 年 / 15 年の
緩和が発散なしに完走することを確認済み)。

なぜ margin = 2 か
------------------
SLD のリミタ ``sld_flux_th`` は境界面の 2 セル先を参照する。赤道の
ゴーストが 1 層だと片側差分に落ち、全球計算と違う拡散フラックスになる
(margin=1 で dq_mt が 9e-5 ずれ、margin=2 で厳密一致)。
"""
from S2MFD.parameters.rempel06 import *   # noqa: F401,F403

import numpy as _np

# --- 計算領域 (Rempel 2005 §2.6) --------------------------------------------
rrmin = 0.65*RSUN          # noqa: F405
rrmax = 0.985*RSUN         # noqa: F405  論文の上端
r_max = 0.985*RSUN         # noqa: F405  Lambda/alpha の基準半径 (= 領域上端)

thmin = 0.0                # 極
thmax = 0.5*_np.pi         # 赤道 (北半球のみ)

# --- 解像度 (Rempel 2005 §2.6: 動径 108 x 緯度 72) ---------------------------
ix = 108
jx = 72
margin = 2                 # SLD のリミタが 2 セル先を見るため

# --- 上端に集中させる 2 点集中格子 -------------------------------------------
# 0.985 RSUN 近傍で密度が急に落ちるので、そこと対流層底に格子を寄せる。
grid_stretch        = [2.0, 3.0]
grid_stretch_center = [0.715*RSUN, 0.985*RSUN]   # noqa: F405
grid_stretch_width  = [0.05*RSUN, 0.03*RSUN]     # noqa: F405
