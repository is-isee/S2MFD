"""人工拡散 (slope-limited diffusion) — 保存形版.

Rempel (2005, 2006) は MacCormack の交互風上/風下差分を使っており, その
スキームに内在する数値散逸で安定性を得ている. 人工粘性を明示的には
入れていない (両論文とも記述なし).

S2MFD は SSP-RK2 + 中心差分で**数値散逸をまったく持たない**ため, 別途
人工拡散が必要になる. これは論文からの逸脱ではなく, 異なる時間積分法を
使うことの必然的な帰結である.

方式は Hotta, Iijima & Kusano (2017) の slope-limited diffusion (SLD).
勾配が格子で解像されている領域では実質ゼロになり, 格子スケールの
振動だけを選択的に潰す.

齋藤 (2024) コードの ``artdif.f90`` が無効化されている理由
-----------------------------------------------------------
齋藤コードにも人工粘性の実装はあるが, ``mhd.f90`` で呼び出しが
コメントアウトされている. 理由は明白で, ``artdif`` が本体と**異なる
保存量** (:math:`r^3\\sin^2\\theta\\,\\Omega_1`, :math:`\\rho_0` を含まない)
に対して定義されており, しかも幾何因子をフラックスの外側に掛けているため,
有効にすると角運動量保存が壊れるからである.

本実装が同じ轍を踏まない設計
----------------------------
**拡散させる対象はプリミティブ変数** (:math:`\\Omega_1`, :math:`v_r`,
:math:`v_\\theta`, :math:`B`) **にし, フラックスは保存量と同じヤコビアンを
掛けた形で作る**. すなわち物理粘性とまったく同じ

.. math::
   F^r = -\\tfrac12 c\\,\\Phi\\,\\rho_0 r^4\\sin^3\\theta\\,
         (\\Omega_{1,i}-\\Omega_{1,i-1})

の形にする (物理粘性の :math:`\\nu\\,\\partial_r\\Omega_1` が
:math:`\\tfrac12 c\\Phi\\,\\Delta\\Omega_1` に置き換わっただけ).
こうすれば

* フラックスは面で 1 回だけ計算され隣接 2 セルで共有される → telescoping
* 境界面はリテラル 0.0 に落とせる → 境界から保存量が漏れない
* 拡散されるのは :math:`\\Omega_1` そのもの

の 3 つが同時に成り立つ.

保存量 :math:`q_L` を直接拡散させないのはなぜか
------------------------------------------------
:math:`q_L=\\rho_0r^4\\sin^3\\theta\\,\\Omega_1` の勾配は, 極近傍では
:math:`\\Omega_1` ではなく幾何因子 :math:`\\sin^3\\theta` の変化に支配される
(第 1 セルと第 2 セルで :math:`\\sin^3\\theta` が 8 倍違う). そこにリミタを
掛けても「解像されていない構造」を検出したことにならず, 一様な
:math:`\\Omega_1` に対してすら拡散フラックスが立ってしまう.

散逸したエネルギーの行き先
--------------------------
人工拡散が奪った運動エネルギー・磁場エネルギーは
:func:`S2MFD.physics.hydro.add_dissipative_heating` でエントロピー方程式に
戻す. 「運動量方程式が奪った分をそのまま渡す」形なので, 離散化によらず
移送は厳密で係数はちょうど 1 になる.
"""
import numpy as np
from numba import njit

from S2MFD.physics.conservative import (
    zero_boundary_faces_r, zero_boundary_faces_th, add_flux_divergence,
)

__all__ = ['sld_flux_r', 'sld_flux_th', 'sld_diffuse']


@njit(inline='always')
def _minmod(a, b):
    if a*b <= 0.0:
        return 0.0
    if abs(a) < abs(b):
        return a
    return b


@njit(fastmath=False)
def sld_flux_r(uu, jac_face, cspeed, coefficient, margin, out):
    """r 方向の slope-limited diffusion フラックスを作る.

    セル境界での「解像されていない跳び」を

    .. math::
       \\Delta_{i-1/2} &= u_i - u_{i-1} \\\\
       \\Psi_i &= {\\rm minmod}(\\Delta_{i-1/2},\\,\\Delta_{i+1/2}) \\\\
       {\\rm jump}_{i-1/2} &= \\Delta_{i-1/2}
         - \\tfrac12(\\Psi_{i-1}+\\Psi_i)

    で測る. 滑らかで解像された場では :math:`\\Psi\\simeq\\Delta` となり
    jump はほぼゼロになる. 格子スケールの振動があるときだけ有限になる.

    フラックスは

    .. math:: F_{i-1/2} = -\\tfrac12\\,h\\,c_{i-1/2}\\,J_{i-1/2}\\,{\\rm jump}

    符号ガード
    ----------
    jump が :math:`\\Delta` と逆符号のときはフラックスをゼロにし,
    :math:`|{\\rm jump}|\\le|\\Delta|` に制限する. これにより拡散は必ず
    勾配を下る向きになり, **散逸が正値であることが構造的に保証される**
    (反拡散になってエネルギーを注入することがない).

    Parameters
    ----------
    uu : numpy.ndarray
        拡散させるプリミティブ変数 (Omega1, v_r, v_theta, B など).
    jac_face : numpy.ndarray
        r 面上のヤコビアン係数 (ixg, jxg). 角運動量なら
        :math:`\\rho_0r^4\\sin^3\\theta`, 運動量なら
        :math:`\\rho_0r^2\\sin\\theta`.
    cspeed : numpy.ndarray
        面上の特性速度 (ixg, jxg). 通常 :math:`|v|+c_{s,\\rm eff}`.
    coefficient : float
        全体に掛かる無次元係数 (``cfg.sld_coefficient``).
    out : numpy.ndarray
        フラックスの出力先 (ixg, jxg). 面 ``i`` はセル ``i-1`` と ``i`` の境界.
    """
    ixg, jxg = uu.shape
    for j in range(jxg):
        for i in range(margin, ixg - margin + 1):
            d0 = uu[i, j] - uu[i - 1, j]
            # 隣接する面差分 (端では自分自身で代用)
            dm = uu[i - 1, j] - uu[i - 2, j] if i - 2 >= 0 else d0
            dp = uu[i + 1, j] - uu[i, j] if i + 1 < ixg else d0
            psi_l = _minmod(dm, d0)
            psi_r = _minmod(d0, dp)
            jump = d0 - 0.5*(psi_l + psi_r)
            # 符号ガード: 勾配を下る向きにだけ効かせる
            if jump*d0 < 0.0:
                jump = 0.0
            elif abs(jump) > abs(d0):
                jump = d0
            out[i, j] = -0.5*coefficient*cspeed[i, j]*jac_face[i, j]*jump


@njit(fastmath=False)
def sld_flux_th(uu, jac_face, cspeed, coefficient, margin, out):
    """theta 方向の slope-limited diffusion フラックス.

    :func:`sld_flux_r` の theta 版. 面 ``j`` はセル ``j-1`` と ``j`` の境界.
    """
    ixg, jxg = uu.shape
    for i in range(ixg):
        for j in range(margin, jxg - margin + 1):
            d0 = uu[i, j] - uu[i, j - 1]
            dm = uu[i, j - 1] - uu[i, j - 2] if j - 2 >= 0 else d0
            dp = uu[i, j + 1] - uu[i, j] if j + 1 < jxg else d0
            psi_l = _minmod(dm, d0)
            psi_r = _minmod(d0, dp)
            jump = d0 - 0.5*(psi_l + psi_r)
            if jump*d0 < 0.0:
                jump = 0.0
            elif abs(jump) > abs(d0):
                jump = d0
            out[i, j] = -0.5*coefficient*cspeed[i, j]*jac_face[i, j]*jump


@njit(fastmath=False)
def sld_diffuse(dqq, uu, jac_r, jac_th, csp_r, csp_th, coefficient,
                drr, dth, margin, ffr, ffth):
    """プリミティブ変数 ``uu`` に人工拡散を掛け, 保存量の時間微分に加算する.

    境界面のフラックスはリテラル 0.0 に落とすので, 人工拡散を通じて
    保存量が境界から出入りすることはない (ユーザー要求
    「境界から保存量が出ていかないように。人工粘性も物理も」).
    """
    sld_flux_r(uu, jac_r, csp_r, coefficient, margin, ffr)
    zero_boundary_faces_r(ffr, margin)
    sld_flux_th(uu, jac_th, csp_th, coefficient, margin, ffth)
    zero_boundary_faces_th(ffth, margin)
    add_flux_divergence(dqq, ffr, ffth, drr, dth, margin)
