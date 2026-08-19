"""保存形の平均場流体ソルバ (Rempel 2006).

:mod:`S2MFD.physics.conservative` の面フラックス演算子の上に, 質量・角運動量
・運動量・エントロピーの発展方程式を組み立てる.

保存量
------
セル体積のヤコビアン :math:`r^2\\sin\\theta` を吸収した形で持つ.

=========================  ==========================================
保存量                     定義
=========================  ==========================================
``q_ro`` (質量)            :math:`r^2\\sin\\theta\\,\\rho_1`
``q_mr`` (動径運動量)      :math:`r^2\\sin\\theta\\,\\rho_0 v_r`
``q_mt`` (子午面運動量)    :math:`r^2\\sin\\theta\\,\\rho_0 v_\\theta`
``q_om`` (角運動量)        :math:`r^4\\sin^3\\theta\\,\\rho_0\\Omega_1`
``q_se`` (エントロピー)    :math:`r^2\\sin\\theta\\,\\rho_0 s_1`
=========================  ==========================================

**厳密な保存が要求されるのは質量と角運動量だけ**である. 球座標では動径・
子午面方向の運動量は幾何学的な源項 (遠心力・曲率) を必然的に持つため
保存量ではない. 回転軸まわりの角運動量が保存するのは系が軸対称だからで,
これは離散化でも厳密に保てる.

角運動量の保存量に :math:`\\Omega_1` だけを入れる理由
------------------------------------------------------
:math:`\\Omega_0` は定数なので

.. math::
   \\frac{\\partial}{\\partial t}\\Big[\\rho_0 r^4\\sin^3\\theta(\\Omega_0+\\Omega_1)\\Big]
   = \\frac{\\partial}{\\partial t}\\Big[\\rho_0 r^4\\sin^3\\theta\\,\\Omega_1\\Big]

であり, 保存量に :math:`\\Omega_0` を含めるかどうかは**解析的には等価**である.
しかし数値的には決定的に違う. 保存量に :math:`\\Omega_0` を含めると, 大きさ
:math:`\\propto\\Omega_0` の数に対して :math:`\\propto\\Omega_1` の増分を毎ステップ
加えることになり, :math:`\\Omega_1` の相対精度は
:math:`\\epsilon\\,\\Omega_0/\\Omega_1` に落ちる. 剛体回転から出発する場合
:math:`\\Omega_1/\\Omega_0\\to0` なので初期ほど悪く, 極端な場合は増分が丸めで
完全に消える. 一方でフラックスは :math:`(\\Omega_0+\\Omega_1)` を運ぶ必要がある
(背景角運動量も子午面循環で運ばれるため).

この使い分けは齋藤 (2024) のコードに由来する
(``mhd.f90`` の ``qqt4`` と ``qqx4`` の違い). 数値検証は
``tests/test_conservation.py`` と ``doc/dev_records`` を参照.
"""
import numpy as np
from numba import njit

from S2MFD.physics.conservative import (
    face_average_r, face_average_th,
    zero_boundary_faces_r, zero_boundary_faces_th,
    add_flux_divergence,
)

__all__ = [
    'HydroWork', 'cfl_dt',
    'mass_rhs', 'angular_momentum_rhs',
    'to_primitive_om1', 'to_primitive_ro1',
]

FOUR_PI = 4.0*np.pi


class HydroWork:
    """時間積分ループ中の一時配列をまとめて確保しておく作業領域.

    毎 substep で ``np.zeros`` を呼ぶと確保と初期化のコストが無視できない
    ため, あらかじめ確保して使い回す.
    """

    def __init__(self, grid):
        shape = (grid.ixg, grid.jxg)
        self.ffr = np.zeros(shape)
        self.ffth = np.zeros(shape)
        self.cen = np.zeros(shape)
        self.cen2 = np.zeros(shape)


@njit(fastmath=False)
def _fill(arr, value):
    ixg, jxg = arr.shape
    for i in range(ixg):
        for j in range(jxg):
            arr[i, j] = value


# ---------------------------------------------------------------------------
# CFL 条件
# ---------------------------------------------------------------------------
@njit(fastmath=False)
def cfl_dt(vrr, vth, brr, bth, bph, ro0, ro1, cs_eff, rr, drr, dth,
           diffusivity, margin, safety, magnetic):
    """許容タイムステップを返す.

    特性速度は**流れ・音速・アルヴェン速度の 3 つ**で決まる:

    .. math::
       \\Delta t = C\\,\\min_{i,j}
         \\frac{\\min(\\Delta r,\\, r\\Delta\\theta)}
              {|\\boldsymbol{v}| + \\sqrt{c_{s,\\rm eff}^2 + v_A^2}}

    音速抑制法 (RSST) が遅くするのは**音波だけ**である. 連続の式に
    :math:`\\xi_s^{-2}` を入れても運動方程式と誘導方程式は変更されないため,
    アルヴェン速度 :math:`v_A = |\\boldsymbol{B}|/\\sqrt{4\\pi\\rho}` は
    そのまま残る. Rempel (2006) のように対流層上部で :math:`\\rho_0` が
    小さく :math:`B_\\varphi` が :math:`10^4` G に達する場合,
    :math:`\\xi_s` をいくら大きくしてもアルヴェン速度が
    :math:`\\Delta t` の下限を決める.

    拡散項 (粘性・磁気拡散) の安定条件
    :math:`\\Delta t < 0.5\\,\\Delta l^2/\\kappa` も併せて課す.

    Parameters
    ----------
    ro1 : numpy.ndarray
        密度摂動. アルヴェン速度は全密度 :math:`\\rho_0+\\rho_1` で評価する.
    cs_eff : numpy.ndarray
        RSST 適用後の実効音速の動径分布 (``ixg``,).
    diffusivity : float
        粘性係数と磁気拡散係数の大きい方 [cm^2/s].
    magnetic : bool
        False ならアルヴェン速度を評価しない.
    """
    ixg, jxg = vrr.shape
    dt = 1.0e30
    for i in range(margin, ixg - margin):
        dl = drr if drr < rr[i]*dth else rr[i]*dth
        cs2 = cs_eff[i]*cs_eff[i]
        for j in range(margin, jxg - margin):
            vv = np.sqrt(vrr[i, j]*vrr[i, j] + vth[i, j]*vth[i, j])
            va2 = 0.0
            if magnetic:
                b2 = (brr[i, j]*brr[i, j] + bth[i, j]*bth[i, j]
                      + bph[i, j]*bph[i, j])
                rho = ro0[i] + ro1[i, j]
                if rho > 0.0:
                    va2 = b2/(FOUR_PI*rho)
            local = dl/(vv + np.sqrt(cs2 + va2))
            if local < dt:
                dt = local
    dt *= safety
    # 拡散の安定条件
    if diffusivity > 0.0:
        dl_min = 1.0e30
        for i in range(margin, ixg - margin):
            dl = drr if drr < rr[i]*dth else rr[i]*dth
            if dl < dl_min:
                dl_min = dl
        dt_diff = 0.5*dl_min*dl_min/diffusivity
        if dt_diff < dt:
            dt = dt_diff
    return dt


# ---------------------------------------------------------------------------
# 質量保存 (音速抑制法)
# ---------------------------------------------------------------------------
@njit(fastmath=False)
def mass_rhs(dq_ro, vrr, vth, JV, JVY, izeta2, drr, dth, margin, ffr, ffth, cen):
    """連続の式の右辺を ``dq_ro`` に加算する.

    .. math::
       \\frac{\\partial}{\\partial t}\\left(r^2\\sin\\theta\\,\\rho_1\\right)
       = -\\frac{1}{\\xi_s^2}\\left[
           \\partial_r\\!\\left(r^2\\sin\\theta\\,\\rho_0 v_r\\right)
         + \\partial_\\theta\\!\\left(r\\sin\\theta\\,\\rho_0 v_\\theta\\right)\\right]

    音速抑制係数 :math:`\\xi_s` は発散全体に掛かるので, 保存量は
    :math:`\\int\\xi_s^2\\rho_1\\,dV` になる. :math:`\\rho_0` は静的なので
    :math:`\\int(\\rho_0+\\xi_s^2\\rho_1)\\,dV` が保存する.

    Parameters
    ----------
    dq_ro : numpy.ndarray
        時間微分の累積先. その場で更新される.
    vrr, vth : numpy.ndarray
        速度のセル中心値.
    JV : numpy.ndarray
        :math:`r^2\\sin\\theta\\,\\rho_0` (``Stratification.JV``).
    JVY : numpy.ndarray
        :math:`r\\sin\\theta\\,\\rho_0` (``Stratification.JVY``).
    izeta2 : numpy.ndarray
        :math:`\\xi_s^{-2}` の 1 次元動径分布 (``ixg``,).
    ffr, ffth, cen : numpy.ndarray
        作業配列 (``HydroWork``).
    """
    ixg, jxg = dq_ro.shape

    # --- r 方向: セル中心で流束量を作り, 面へ算術平均 ---------------------
    for i in range(ixg):
        for j in range(jxg):
            cen[i, j] = JV[i, j]*vrr[i, j]
    face_average_r(cen, margin, ffr)
    zero_boundary_faces_r(ffr, margin)

    # --- theta 方向 -------------------------------------------------------
    for i in range(ixg):
        for j in range(jxg):
            cen[i, j] = JVY[i, j]*vth[i, j]
    face_average_th(cen, margin, ffth)
    zero_boundary_faces_th(ffth, margin)

    # --- 発散 (RSST 係数は発散全体に掛ける) -------------------------------
    idrr = 1.0/drr
    idth = 1.0/dth
    for i in range(margin, ixg - margin):
        iz2 = izeta2[i]
        for j in range(margin, jxg - margin):
            dq_ro[i, j] -= iz2*((ffr[i + 1, j] - ffr[i, j])*idrr
                                + (ffth[i, j + 1] - ffth[i, j])*idth)


# ---------------------------------------------------------------------------
# 角運動量保存
# ---------------------------------------------------------------------------
@njit(fastmath=False)
def angular_momentum_rhs(dq_om, om1, vrr, vth, brr, bth, bph,
                         JL, JLY, RR, RRm, sinTH, sinTHm, ro0, ro0m,
                         lam_rp, lam_tp, om0, nu, drr, dth, margin,
                         magnetic, ffr, ffth, cen):
    """角運動量方程式の右辺を ``dq_om`` に加算する.

    すべての項を発散形

    .. math::
       \\frac{\\partial q_L}{\\partial t}
       = -\\partial_r F^r - \\partial_\\theta F^\\theta

    で書き, 幾何学的な源項を一切持たない. 各フラックスの寄与:

    移流
        :math:`F^r = r^4\\sin^3\\theta\\,\\rho_0(\\Omega_0+\\Omega_1)v_r`,
        :math:`F^\\theta = r^3\\sin^3\\theta\\,\\rho_0(\\Omega_0+\\Omega_1)v_\\theta`
    粘性
        :math:`F^r = -\\nu\\rho_0 r^4\\sin^3\\theta\\,\\partial_r\\Omega_1`,
        :math:`F^\\theta = -\\nu\\rho_0 r^2\\sin^3\\theta\\,\\partial_\\theta\\Omega_1`
    :math:`\\Lambda` 効果
        :math:`F^r = -\\nu\\rho_0 r^3\\sin^2\\theta\\,\\Lambda_{r\\varphi}`,
        :math:`F^\\theta = -\\nu\\rho_0 r^2\\sin^2\\theta\\,\\Lambda_{\\theta\\varphi}`
    Maxwell 応力
        :math:`F^r = -r^3\\sin^2\\theta\\,B_r B_\\varphi/4\\pi`,
        :math:`F^\\theta = -r^2\\sin^2\\theta\\,B_\\theta B_\\varphi/4\\pi`

    移流フラックスは齋藤 (2024) に合わせ, 積 :math:`q(\\Omega_0+\\Omega_1)v`
    をセル中心で作ってから面へ算術平均する (因子ごとに平均するのではない).
    拡散フラックスは面上の勾配をそのまま使い, 係数だけを面へ平均する.

    Parameters
    ----------
    magnetic : bool
        Maxwell 応力を含めるかどうか. ``dynamics == 'hydro'`` では False.
    ro0m : numpy.ndarray
        :math:`\\rho_0` の r 面上の値 (``ixg``,). 面 ``i`` はセル
        ``i-1`` と ``i`` の境界.
    """
    ixg, jxg = dq_om.shape

    # =====================================================================
    # r 方向フラックス
    # =====================================================================
    # --- 移流: 積を中心で作ってから面平均 --------------------------------
    for i in range(ixg):
        for j in range(jxg):
            cen[i, j] = JL[i, j]*(om0 + om1[i, j])*vrr[i, j]
    face_average_r(cen, margin, ffr)

    # --- 粘性 + Lambda 効果 + Maxwell 応力を同じ面配列に足し込む ---------
    idrr = 1.0/drr
    for i in range(margin, ixg - margin + 1):
        rm = RRm[i, 0]
        rm3 = rm*rm*rm
        rm4 = rm3*rm
        cvis = -nu*ro0m[i]*rm4
        clam = -nu*ro0m[i]*rm3
        for j in range(jxg):
            s = sinTH[i, j]
            s2 = s*s
            s3 = s2*s
            # 粘性: 面上の勾配をそのまま使う (2 セルで共有される)
            ffr[i, j] += cvis*s3*(om1[i, j] - om1[i - 1, j])*idrr
            # Lambda 効果: セル中心の値を面へ平均
            ffr[i, j] += clam*s2*0.5*(lam_rp[i, j] + lam_rp[i - 1, j])
    if magnetic:
        for i in range(ixg):
            for j in range(jxg):
                s = sinTH[i, j]
                r = RR[i, j]
                cen[i, j] = -r*r*r*s*s*brr[i, j]*bph[i, j]/FOUR_PI
        for i in range(margin, ixg - margin + 1):
            for j in range(jxg):
                ffr[i, j] += 0.5*(cen[i, j] + cen[i - 1, j])

    zero_boundary_faces_r(ffr, margin)

    # =====================================================================
    # theta 方向フラックス
    # =====================================================================
    for i in range(ixg):
        for j in range(jxg):
            cen[i, j] = JLY[i, j]*(om0 + om1[i, j])*vth[i, j]
    face_average_th(cen, margin, ffth)

    idth = 1.0/dth
    for i in range(ixg):
        r = RR[i, 0]
        r2 = r*r
        cvis = -nu*ro0[i]*r2
        clam = -nu*ro0[i]*r2
        for j in range(margin, jxg - margin + 1):
            sm = sinTHm[i, j]
            sm2 = sm*sm
            sm3 = sm2*sm
            ffth[i, j] += cvis*sm3*(om1[i, j] - om1[i, j - 1])*idth
            ffth[i, j] += clam*0.5*(sinTH[i, j]*sinTH[i, j]*lam_tp[i, j]
                                    + sinTH[i, j - 1]*sinTH[i, j - 1]*lam_tp[i, j - 1])
    if magnetic:
        for i in range(ixg):
            for j in range(jxg):
                s = sinTH[i, j]
                r = RR[i, j]
                cen[i, j] = -r*r*s*s*bth[i, j]*bph[i, j]/FOUR_PI
        for i in range(ixg):
            for j in range(margin, jxg - margin + 1):
                ffth[i, j] += 0.5*(cen[i, j] + cen[i, j - 1])

    zero_boundary_faces_th(ffth, margin)

    # =====================================================================
    # 発散 (セル体積で割らない — ヤコビアンは保存量に吸収済み)
    # =====================================================================
    add_flux_divergence(dq_om, ffr, ffth, drr, dth, margin)


# ---------------------------------------------------------------------------
# 保存量 <-> プリミティブ変数
# ---------------------------------------------------------------------------
@njit(fastmath=False)
def to_primitive_om1(q_om, iJL, om1, margin):
    """:math:`q_L \\to \\Omega_1`. 物理セルのみ変換する (ゴーストは境界条件で埋める)."""
    ixg, jxg = q_om.shape
    for i in range(margin, ixg - margin):
        for j in range(margin, jxg - margin):
            om1[i, j] = q_om[i, j]*iJL[i, j]


@njit(fastmath=False)
def to_primitive_ro1(q_ro, iJM, ro1, margin):
    """:math:`q_\\rho \\to \\rho_1`. 物理セルのみ変換する."""
    ixg, jxg = q_ro.shape
    for i in range(margin, ixg - margin):
        for j in range(margin, jxg - margin):
            ro1[i, j] = q_ro[i, j]*iJM[i, j]
