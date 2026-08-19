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

なぜ SLD だけでは足りないか (部分的な風上化の下限)
--------------------------------------------------
SLD は「解像された場には効かない」ことが利点だが, 本モデルではそれが
足りない. Rempel の MacCormack (交互風上/風下差分) は音波に対して
:math:`\\tfrac12 c_{s,\\rm eff}\\Delta x \\simeq 5\\times10^{13}\\,
{\\rm cm^2/s}` 相当の数値散逸を持っており, これは物理的な乱流粘性
:math:`\\nu_t=3\\times10^{12}` の 15 倍にもなる. 対流層が中立成層
(:math:`\\delta=0`) の場合, 復元力がないため 6-8 セル程度の**解像された**
モードが準中立になり, SLD ではこれを抑えられずに成長する
(実際に :math:`\\kappa_t` または :math:`\\nu_t` を 10 倍にすると安定化する
ことを確認した).

そこで ``floor`` を導入し, リミタが「解像されている」と判断した場合でも
その割合だけは Rusanov 型の散逸を残す. ``floor=0`` なら純粋な SLD,
``floor=1`` なら完全な Rusanov (1 次風上) になる. Rempel のスキームが
持つ暗黙の散逸に相当する量を明示的に入れるための係数であり,
論文からの逸脱ではなく**異なる時間積分法を使うことの必然的な代償**である.

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
    zero_boundary_faces_r, zero_boundary_faces_th,
    add_flux_divergence, add_flux_divergence_scaled, add_flux_work,
)

__all__ = ['sld_flux_r', 'sld_flux_th', 'sld_diffuse', 'sld_diffuse_work',
           'sld_diffuse_scaled', 'sld_diffuse_primitive', 'sld_diffusivity_max']


@njit(inline='always')
def _minmod(a, b):
    if a*b <= 0.0:
        return 0.0
    if abs(a) < abs(b):
        return a
    return b


@njit(fastmath=False)
def sld_flux_r(uu, jac_face, cspeed, coefficient, floor, margin, out):
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
            # 部分的な風上化 (下限). リミタが「解像されている」と判断した
            # 場合でも floor の割合だけは Rusanov 型の散逸を残す
            if abs(jump) < floor*abs(d0):
                jump = floor*d0
            out[i, j] = -0.5*coefficient*cspeed[i, j]*jac_face[i, j]*jump


@njit(fastmath=False)
def sld_flux_th(uu, jac_face, cspeed, coefficient, floor, margin, out):
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
            # 部分的な風上化 (下限). リミタが「解像されている」と判断した
            # 場合でも floor の割合だけは Rusanov 型の散逸を残す
            if abs(jump) < floor*abs(d0):
                jump = floor*d0
            out[i, j] = -0.5*coefficient*cspeed[i, j]*jac_face[i, j]*jump


@njit(fastmath=False)
def sld_diffuse(dqq, uu, jac_r, jac_th, csp_r, csp_th, coefficient, floor,
                drr, dth, margin, ffr, ffth):
    """プリミティブ変数 ``uu`` に人工拡散を掛け, 保存量の時間微分に加算する.

    境界面のフラックスはリテラル 0.0 に落とすので, 人工拡散を通じて
    保存量が境界から出入りすることはない (ユーザー要求
    「境界から保存量が出ていかないように。人工粘性も物理も」).
    """
    sld_flux_r(uu, jac_r, csp_r, coefficient, floor, margin, ffr)
    zero_boundary_faces_r(ffr, margin)
    sld_flux_th(uu, jac_th, csp_th, coefficient, floor, margin, ffth)
    zero_boundary_faces_th(ffth, margin)
    add_flux_divergence(dqq, ffr, ffth, drr, dth, margin)


@njit(fastmath=False)
def sld_diffuse_scaled(dqq, uu, jac_r, jac_th, csp_r, csp_th, coefficient,
                       floor, drr, dth, margin, scale, ffr, ffth):
    """:func:`sld_diffuse` の, 発散に動径依存の係数が掛かる版.

    密度に人工拡散を掛けるときに使う. 音速抑制法では保存量が
    :math:`\\int\\xi_s^2\\rho_1\\,dV` なので, 連続の式と同じく
    :math:`\\xi_s^{-2}` を発散全体に掛けないと質量保存が壊れる
    (:math:`\\xi_s` が動径に依存する場合).
    """
    sld_flux_r(uu, jac_r, csp_r, coefficient, floor, margin, ffr)
    zero_boundary_faces_r(ffr, margin)
    sld_flux_th(uu, jac_th, csp_th, coefficient, floor, margin, ffth)
    zero_boundary_faces_th(ffth, margin)
    add_flux_divergence_scaled(dqq, ffr, ffth, drr, dth, margin, scale)


@njit(fastmath=False)
def sld_diffuse_primitive(duu, uu, jac_r, jac_th, ijac, csp_r, csp_th,
                          coefficient, floor, drr, dth, margin, ffr, ffth):
    """保存形ではなく直接解いているプリミティブ変数に人工拡散を掛ける.

    エントロピーのように保存量として持っていない変数に使う.
    フラックスは保存形と同じヤコビアン付きで作り, 最後にセル中心の
    ヤコビアンで割って :math:`\\partial u/\\partial t` に変換する.

    エントロピーへの人工拡散を忘れると何が起きるか
    ----------------------------------------------
    :math:`s_1` は中心差分で移流されるだけで散逸を持たないため, 格子スケールの
    ノイズが減衰しない. そのノイズは状態方程式
    :math:`p_1=p_0(\\gamma\\rho_1/\\rho_0+s_1)` を通して圧力に乗り,
    :math:`\\rho_0` が小さい対流層上部で
    :math:`\\rho_0^{-1}\\partial_r p_1` として巨大な加速に化ける. その速度が
    さらに :math:`s_1` をかき混ぜるので正のフィードバックになり,
    計算が発散する. 実際に赤道近傍・上部対流層から壊れることを確認した
    (``doc/dev_records`` の作業記録を参照).
    """
    sld_flux_r(uu, jac_r, csp_r, coefficient, floor, margin, ffr)
    zero_boundary_faces_r(ffr, margin)
    sld_flux_th(uu, jac_th, csp_th, coefficient, floor, margin, ffth)
    zero_boundary_faces_th(ffth, margin)
    ixg, jxg = duu.shape
    idrr = 1.0/drr
    idth = 1.0/dth
    for i in range(margin, ixg - margin):
        for j in range(margin, jxg - margin):
            duu[i, j] -= ((ffr[i + 1, j] - ffr[i, j])*idrr
                          + (ffth[i, j + 1] - ffth[i, j])*idth)*ijac[i, j]


@njit(fastmath=False)
def sld_diffusivity_max(csp_r, csp_th, coefficient, floor, drr, dth, rr, margin):
    """人工拡散の実効拡散係数の最大値を返す (CFL 用).

    SLD フラックスは :math:`-\\tfrac12 h c\\,\\Delta u` なので, 実効的な
    拡散係数は :math:`\\tfrac12 h c\\,\\Delta x` に相当する. これを
    陽解法の拡散安定条件に入れないと, ``sld_coefficient`` を大きくしたときに
    静かに不安定化する (係数 4 で即座に発散することを確認済み).
    """
    ixg, jxg = csp_r.shape
    kmax = 0.0
    for i in range(margin, ixg - margin):
        dl = drr if drr < rr[i]*dth else rr[i]*dth
        for j in range(margin, jxg - margin):
            c = csp_r[i, j] if csp_r[i, j] > csp_th[i, j] else csp_th[i, j]
            k = 0.5*coefficient*c*dl
            if k > kmax:
                kmax = k
    return kmax


@njit(fastmath=False)
def sld_diffuse_work(dqq, heat, uu, jac_r, jac_th, csp_r, csp_th, coefficient,
                     floor, drr, dth, margin, ffr, ffth):
    """:func:`sld_diffuse` に加えて, 局所的な散逸率を ``heat`` に積む.

    角運動量に人工拡散を掛けるときはこちらを使う.
    :math:`\\Omega=\\Omega_0+\\Omega_1` の :math:`\\Omega_0` は
    :math:`-F\\cdot\\nabla\\Omega` の微分で消えるので, 局所的な加熱が
    :math:`\\Omega_0/\\Omega_1` 倍に化ける問題を避けられる
    (:func:`S2MFD.physics.conservative.add_flux_work` の説明を参照).
    """
    sld_flux_r(uu, jac_r, csp_r, coefficient, floor, margin, ffr)
    zero_boundary_faces_r(ffr, margin)
    sld_flux_th(uu, jac_th, csp_th, coefficient, floor, margin, ffth)
    zero_boundary_faces_th(ffth, margin)
    add_flux_divergence(dqq, ffr, ffth, drr, dth, margin)
    add_flux_work(heat, ffr, ffth, uu, drr, dth, margin)
