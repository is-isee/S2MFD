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

from ._jit import kernel, prange

from S2MFD.physics.conservative import (
    face_average_r, face_average_th,
    zero_boundary_faces_r, zero_boundary_faces_th,
    add_flux_divergence, add_flux_work,
)

__all__ = [
    'HydroWork', 'cfl_dt',
    'mass_rhs', 'angular_momentum_rhs', 'momentum_rhs',
    'viscous_meridional_rhs',
    'entropy_rhs', 'add_dissipative_heating', 'add_ohmic_heating',
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
        # 子午面速度の人工拡散は 2 成分を同時に扱うので面配列がもう 1 組要る
        self.ffr2 = np.zeros(shape)
        self.ffth2 = np.zeros(shape)


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
        dl = drr[i] if drr[i] < rr[i]*dth else rr[i]*dth
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
            dl = drr[i] if drr[i] < rr[i]*dth else rr[i]*dth
            if dl < dl_min:
                dl_min = dl
        # 安全率は移流側と同じものを掛ける。ここを忘れると拡散が
        # ぎりぎり安定な設定で静かに壊れる
        dt_diff = safety*0.5*dl_min*dl_min/diffusivity
        if dt_diff < dt:
            dt = dt_diff
    return dt


# ---------------------------------------------------------------------------
# 質量保存 (音速抑制法)
# ---------------------------------------------------------------------------
@kernel()
def mass_rhs(dq_ro, vrr, vth, JV, JVY, izeta2, drr, wfm, dth, margin,
             ffr, ffth, cen):
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
    for i in prange(ixg):
        for j in range(jxg):
            cen[i, j] = JV[i, j]*vrr[i, j]
    face_average_r(cen, wfm, margin, ffr)
    zero_boundary_faces_r(ffr, margin)

    # --- theta 方向 -------------------------------------------------------
    for i in prange(ixg):
        for j in range(jxg):
            cen[i, j] = JVY[i, j]*vth[i, j]
    face_average_th(cen, margin, ffth)
    zero_boundary_faces_th(ffth, margin)

    # --- 発散 (RSST 係数は発散全体に掛ける) -------------------------------
    idth = 1.0/dth
    for i in prange(margin, ixg - margin):
        idrr = 1.0/drr[i]
        iz2 = izeta2[i]
        for j in range(margin, jxg - margin):
            dq_ro[i, j] -= iz2*((ffr[i + 1, j] - ffr[i, j])*idrr
                                + (ffth[i, j + 1] - ffth[i, j])*idth)


# ---------------------------------------------------------------------------
# 角運動量保存
# ---------------------------------------------------------------------------
@kernel()
def angular_momentum_rhs(dq_om, om1, vrr, vth, brr, bth, bph,
                         JL, JLY, JV, JVY, W2, RR, RRm, sinTH, sinTHm,
                         ro0, ro0m, lam_rp, lam_tp, om0,
                         nu_dif, nu_dif_m, nu_lam, nu_lam_m,
                         drr, drrm, wfm, dth, margin,
                         magnetic, consistent_advection, open_bottom,
                         dq_stress, heat, bflux, ffr, ffth, cen):
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

    出力の分け方
    ------------
    ``dq_om`` には移流と Maxwell 応力だけを加算し, **レイノルズ応力
    (粘性 + :math:`\\Lambda` 効果) は ``dq_stress`` にだけ**書き出す.
    呼び出し側で ``dq_om += dq_stress`` すること. 分けているのは
    エントロピー方程式の加熱項 :math:`Q=\\sum\\tfrac12 E_{ik}R_{ik}` が
    「レイノルズ応力が流れにした仕事の符号反転」に等しく,
    :func:`add_dissipative_heating` にそのまま渡せるようにするため.

    Parameters
    ----------
    magnetic : bool
        Maxwell 応力を含めるかどうか. ``dynamics == 'hydro'`` では False.
    consistent_advection : bool
        移流フラックスの作り方を選ぶ.

        False (既定, 齋藤 2024 と同じ)
            積 :math:`q_L(\\Omega_0+\\Omega_1)v` をセル中心で作ってから
            面へ算術平均する.
        True
            連続の式で使うのと**同一の質量流束の面値**に, 比角運動量
            :math:`\\varpi^2(\\Omega_0+\\Omega_1)` の面値を掛ける.

        どちらも面値を 2 セルで共有するので保存は厳密に成り立つ. 違うのは
        精度で, True のほうが「質量方程式との整合性」が良い.

        なぜ気にするか: 保存形は移流形 :math:`\\rho_0 D(\\varpi^2\\Omega)/Dt`
        に対して :math:`-\\Omega_0\\varpi^2\\nabla\\cdot(\\rho_0\\boldsymbol v)`
        だけ余分な項を持つ. 質量流束が離散的に非発散なら消えるが, 音速抑制法
        では :math:`\\nabla\\cdot(\\rho_0\\boldsymbol v)=-\\xi_s^2
        \\partial\\rho_1/\\partial t` なので厳密にはゼロにならず, しかも係数
        :math:`\\Omega_0` で増幅される. True にすると角運動量フラックスが
        質量流束そのものに比例するので, :math:`\\Omega` が一様なときに
        角運動量方程式が質量方程式に帰着する (free-stream preservation).
    ro0m : numpy.ndarray
        :math:`\\rho_0` の r 面上の値 (``ixg``,). 面 ``i`` はセル
        ``i-1`` と ``i`` の境界.
    W2 : numpy.ndarray
        :math:`\\varpi^2 = r^2\\sin^2\\theta` (``Stratification.W2``).
    """
    ixg, jxg = dq_om.shape

    # =====================================================================
    # r 方向フラックス
    # =====================================================================
    # --- 移流 -------------------------------------------------------------
    if consistent_advection:
        # 質量流束と同一の面値を使い、それに比角運動量の面値を掛ける。
        # F^r = M^r_face * (varpi^2 (Om0+Om1))_face
        # 連続の式で使う M^r と同じ配列なので、Omega が一様なら角運動量
        # 方程式は質量方程式にそのまま帰着する (free-stream preservation)。
        for i in range(ixg):
            for j in range(jxg):
                cen[i, j] = JV[i, j]*vrr[i, j]
        face_average_r(cen, wfm, margin, ffr)
        for i in range(ixg):
            for j in range(jxg):
                cen[i, j] = W2[i, j]*(om0 + om1[i, j])
        for i in range(margin, ixg - margin + 1):
            w = wfm[i]
            w1 = 1.0 - w
            for j in range(jxg):
                ffr[i, j] *= w*cen[i - 1, j] + w1*cen[i, j]
    else:
        # 齋藤 (2024) と同じ: 積をセル中心で作ってから面平均する
        for i in range(ixg):
            for j in range(jxg):
                cen[i, j] = JL[i, j]*(om0 + om1[i, j])*vrr[i, j]
        face_average_r(cen, wfm, margin, ffr)

    # --- 粘性 + Lambda 効果 + Maxwell 応力を同じ面配列に足し込む ---------
    if magnetic:
        for i in range(ixg):
            for j in range(jxg):
                s = sinTH[i, j]
                r = RR[i, j]
                cen[i, j] = -r*r*r*s*s*brr[i, j]*bph[i, j]/FOUR_PI
        for i in range(margin, ixg - margin + 1):
            w = wfm[i]
            w1 = 1.0 - w
            for j in range(jxg):
                ffr[i, j] += w*cen[i - 1, j] + w1*cen[i, j]

    zero_boundary_faces_r(ffr, margin)

    # =====================================================================
    # theta 方向フラックス (移流 + Maxwell)
    # =====================================================================
    if consistent_advection:
        for i in range(ixg):
            for j in range(jxg):
                cen[i, j] = JVY[i, j]*vth[i, j]
        face_average_th(cen, margin, ffth)
        for i in range(ixg):
            for j in range(jxg):
                cen[i, j] = W2[i, j]*(om0 + om1[i, j])
        for i in range(ixg):
            for j in range(margin, jxg - margin + 1):
                ffth[i, j] *= 0.5*(cen[i, j] + cen[i, j - 1])
    else:
        for i in range(ixg):
            for j in range(jxg):
                cen[i, j] = JLY[i, j]*(om0 + om1[i, j])*vth[i, j]
        face_average_th(cen, margin, ffth)

    idth = 1.0/dth
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

    # 発散 (セル体積で割らない — ヤコビアンは保存量に吸収済み)
    add_flux_divergence(dq_om, ffr, ffth, drr, dth, margin)

    # =====================================================================
    # レイノルズ応力 (粘性 + Lambda 効果)
    # ---------------------------------------------------------------------
    # 移流や Maxwell 応力とは別の配列にも発散を書き出す. エントロピー方程式の
    # 加熱項 Q = sum (1/2) E_ik R_ik は「レイノルズ応力が流れにした仕事の
    # 符号反転」に等しい (境界フラックスがゼロなので部分積分の表面項が消える)
    # ので, ここで分離しておけば add_dissipative_heating に渡すだけでよい.
    # 粘性項は正の加熱, Lambda 効果は負の加熱 (差動回転にエネルギーを渡す側)
    # になり, Rempel 2005 の記述「Q は粘性散逸による加熱項と Lambda 効果に
    # よる冷却項を含み, 後者が一般に支配的」がそのまま再現される.
    # =====================================================================
    for i in prange(margin, ixg - margin + 1):
        rm = RRm[i, 0]
        rm3 = rm*rm*rm
        rm4 = rm3*rm
        cvis = -nu_dif_m[i]*ro0m[i]*rm4
        clam = -nu_lam_m[i]*ro0m[i]*rm3
        # 面での勾配なので **セル中心間距離** で割る (セル幅ではない)
        idrrm = 1.0/drrm[i]
        w = wfm[i]
        w1 = 1.0 - w
        for j in range(jxg):
            sn = sinTH[i, j]
            sn2 = sn*sn
            sn3 = sn2*sn
            # 粘性: 面上の勾配をそのまま使う (2 セルで共有される)
            # Lambda 効果: セル中心の値を面へ平均
            ffr[i, j] = (cvis*sn3*(om1[i, j] - om1[i - 1, j])*idrrm
                         + clam*sn2*(w*lam_rp[i - 1, j] + w1*lam_rp[i, j]))
    if open_bottom:
        # Rempel 2005/2006 の下部境界 (Omega1 = 0 の剛体回転リザーバ) では
        # 粘性フラックスが境界を通る。これがタコクラインを形成するトルクで、
        # 系は角運動量について閉じなくなる。フラックスをゼロにすると
        # 境界条件が実質無効になるので、粘性分だけ残す。
        # Lambda 効果は内部の再分配なので境界では落とす。
        for j in range(jxg):
            rm = RRm[margin, 0]
            rm4 = rm*rm*rm*rm
            sn = sinTH[margin, j]
            bflux[j] = (-nu_dif_m[margin]*ro0m[margin]*rm4*sn*sn*sn
                        * (om1[margin, j] - om1[margin - 1, j])/drrm[margin])
            ffr[margin, j] = bflux[j]
            ffr[ixg - margin, j] = 0.0
    else:
        zero_boundary_faces_r(ffr, margin)
        for j in range(jxg):
            bflux[j] = 0.0

    for i in prange(ixg):
        r = RR[i, 0]
        r2 = r*r
        cvis = -nu_dif[i]*ro0[i]*r2
        clam = -nu_lam[i]*ro0[i]*r2
        for j in range(margin, jxg - margin + 1):
            sm = sinTHm[i, j]
            sm3 = sm*sm*sm
            ffth[i, j] = (cvis*sm3*(om1[i, j] - om1[i, j - 1])*idth
                          + clam*0.5*(sinTH[i, j]*sinTH[i, j]*lam_tp[i, j]
                                      + sinTH[i, j - 1]*sinTH[i, j - 1]*lam_tp[i, j - 1]))
    zero_boundary_faces_th(ffth, margin)

    # レイノルズ応力は dq_stress にだけ書き出す。dq_om への合流は呼び出し側が
    # 行う (dq_stress はエントロピーの加熱項にも使うため分けている)。
    # ここで dq_om にも足すと二重計上になり、粘性と Lambda 効果が 2 倍になる。
    add_flux_divergence(dq_stress, ffr, ffth, drr, dth, margin)
    # 局所的なエネルギー変換率 -F.grad(Omega1) をエントロピーの加熱項へ。
    # Omega0 は微分で消えるので、Omega0/Omega1 倍の偽の加熱が生じない。
    add_flux_work(heat, ffr, ffth, om1, drr, dth, margin)


# ---------------------------------------------------------------------------
# 子午面の運動量 (r, theta 成分)
# ---------------------------------------------------------------------------
@kernel()
def momentum_rhs(dq_mr, dq_mt, vrr, vth, om1, ro1, pr1, brr, bth, bph,
                 JV, JVY, JM, RSIN, RR, sinTH, cosTH, ro0, gr,
                 om0, drr, drr2, wfm, dth, margin, magnetic, ffr, ffth, cen,
                 magnetic_buoyancy=True):
    """動径・子午面運動量の右辺を加算する.

    .. math::
       \\frac{\\partial}{\\partial t}\\left(r^2\\sin\\theta\\,\\rho_0 v_r\\right)
       &= -\\partial_r\\!\\left(r^2\\sin\\theta\\,\\rho_0 v_r v_r\\right)
          -\\partial_\\theta\\!\\left(r\\sin\\theta\\,\\rho_0 v_r v_\\theta\\right)\\\\
       &\\quad + r\\sin\\theta\\,\\rho_0\\!\\left[v_\\theta^2
              + r^2\\sin^2\\theta\\,(2\\Omega_0\\Omega_1+\\Omega_1^2)\\right]
          - r^2\\sin\\theta\\left(\\rho_1 g + \\partial_r p_1\\right)
          + r^2\\sin\\theta\\,F_{L,r}

    .. math::
       \\frac{\\partial}{\\partial t}\\left(r^2\\sin\\theta\\,\\rho_0 v_\\theta\\right)
       &= -\\partial_r\\!\\left(r^2\\sin\\theta\\,\\rho_0 v_\\theta v_r\\right)
          -\\partial_\\theta\\!\\left(r\\sin\\theta\\,\\rho_0 v_\\theta v_\\theta\\right)\\\\
       &\\quad - r\\sin\\theta\\,\\rho_0 v_r v_\\theta
          + r^3\\sin^2\\theta\\cos\\theta\\,\\rho_0(2\\Omega_0\\Omega_1+\\Omega_1^2)
          - r\\sin\\theta\\,\\partial_\\theta p_1
          + r^2\\sin\\theta\\,F_{L,\\theta}

    **子午面の運動量は保存量ではない.** 球座標では曲率と遠心力に由来する
    幾何学的な源項が必ず現れるためで, これは離散化の不備ではなく物理である
    (軸対称系で厳密に保存するのは回転軸まわりの角運動量だけ).
    したがってここでは源項を持つ形をそのまま使う.

    遠心力の摂動形
    --------------
    遠心力は :math:`(\\Omega_0+\\Omega_1)^2` に比例するが, ここでは
    :math:`2\\Omega_0\\Omega_1+\\Omega_1^2` すなわち
    :math:`(\\Omega_0+\\Omega_1)^2-\\Omega_0^2` を使う. 剛体回転部分
    :math:`\\Omega_0^2` の遠心力は背景の釣り合いに含まれているためで,
    角運動量の保存量から :math:`\\Omega_0` を外すのと同じ「摂動だけを持つ」
    方針である. これにより :math:`\\Omega_1\\ll\\Omega_0` のときの桁落ちを
    避けられる.

    圧力勾配は面平均した :math:`p_1` の差分で評価する (齋藤 2024 と同じ).
    背景の静水圧平衡は解析的に差し引かれているので, ここに現れるのは
    摂動だけである (well-balanced).
    """
    ixg, jxg = dq_mr.shape
    idth = 1.0/dth

    # =====================================================================
    # 移流: r 方向運動量
    # =====================================================================
    for i in prange(ixg):
        for j in range(jxg):
            cen[i, j] = JV[i, j]*vrr[i, j]*vrr[i, j]
    face_average_r(cen, wfm, margin, ffr)
    zero_boundary_faces_r(ffr, margin)
    for i in prange(ixg):
        for j in range(jxg):
            cen[i, j] = JVY[i, j]*vrr[i, j]*vth[i, j]
    face_average_th(cen, margin, ffth)
    zero_boundary_faces_th(ffth, margin)
    add_flux_divergence(dq_mr, ffr, ffth, drr, dth, margin)

    # =====================================================================
    # 移流: theta 方向運動量
    # =====================================================================
    for i in prange(ixg):
        for j in range(jxg):
            cen[i, j] = JV[i, j]*vth[i, j]*vrr[i, j]
    face_average_r(cen, wfm, margin, ffr)
    zero_boundary_faces_r(ffr, margin)
    for i in prange(ixg):
        for j in range(jxg):
            cen[i, j] = JVY[i, j]*vth[i, j]*vth[i, j]
    face_average_th(cen, margin, ffth)
    zero_boundary_faces_th(ffth, margin)
    add_flux_divergence(dq_mt, ffr, ffth, drr, dth, margin)

    # =====================================================================
    # 幾何学的源項 + 圧力勾配 + 浮力
    # =====================================================================
    for i in prange(margin, ixg - margin):
        # セル中心の 2 セル幅中心差分なので rr[i+1]-rr[i-1] で割る
        idrr2 = 1.0/drr2[i]
        for j in range(margin, jxg - margin):
            o1 = om1[i, j]
            cent = 2.0*om0*o1 + o1*o1          # (Om0+Om1)^2 - Om0^2
            rs = RSIN[i, j]
            r = RR[i, j]
            vt = vth[i, j]

            # 圧力勾配 (面平均した p1 の差分 = 2 セル幅の中心差分)
            dprr = (pr1[i + 1, j] - pr1[i - 1, j])*idrr2
            dprt = 0.5*(pr1[i, j + 1] - pr1[i, j - 1])*idth

            dq_mr[i, j] += (JVY[i, j]*(vt*vt + rs*rs*cent)
                            - JM[i, j]*(ro1[i, j]*gr[i] + dprr))
            dq_mt[i, j] += (-JVY[i, j]*vrr[i, j]*vt
                            + JVY[i, j]*r*r*sinTH[i, j]*cosTH[i, j]*cent
                            - rs*dprt)

    # =====================================================================
    # ローレンツ力 (子午面成分)
    # =====================================================================
    if magnetic:
        for i in range(margin, ixg - margin):
            idrr2 = 1.0/drr2[i]
            for j in range(margin, jxg - margin):
                r = RR[i, j]
                s = sinTH[i, j]
                # J = rot(B)
                jr = (s*0.0 + (sinTH[i, j + 1]*bph[i, j + 1]
                               - sinTH[i, j - 1]*bph[i, j - 1])
                      * 0.5*idth)/(r*s)
                jt = -(RR[i + 1, j]*bph[i + 1, j]
                       - RR[i - 1, j]*bph[i - 1, j])*idrr2/r
                jp = ((RR[i + 1, j]*bth[i + 1, j]
                       - RR[i - 1, j]*bth[i - 1, j])*idrr2
                      - (brr[i, j + 1] - brr[i, j - 1])*0.5*idth)/r
                fl_r = (jt*bph[i, j] - jp*bth[i, j])/FOUR_PI
                fl_t = (jp*brr[i, j] - jr*bph[i, j])/FOUR_PI
                if not magnetic_buoyancy:
                    # Rempel (2006) 表1 の "magnetic buoyancy off" (列 4,6,8):
                    # 式 (2) の grad p_mag を落とす。J x B = (B.grad)B/4pi
                    # - grad(B^2/8pi) なので、動径方向の磁気圧勾配を足し戻せば
                    # 磁気張力だけが残る。軸対称モデルの磁気浮力は非現実的
                    # (実際の浮力不安定は非軸対称) というのが論文の理由付け。
                    bsq_p = (brr[i + 1, j]**2 + bth[i + 1, j]**2
                             + bph[i + 1, j]**2)
                    bsq_m = (brr[i - 1, j]**2 + bth[i - 1, j]**2
                             + bph[i - 1, j]**2)
                    fl_r += (bsq_p - bsq_m)*idrr2/(2.0*FOUR_PI)
                dq_mr[i, j] += JM[i, j]*fl_r
                dq_mt[i, j] += JM[i, j]*fl_t


# ---------------------------------------------------------------------------
# 子午面の粘性応力
# ---------------------------------------------------------------------------
@kernel()
def viscous_meridional_rhs(dq_mr, dq_mt, vrr, vth, rr, sinTH, cosTH, ro0,
                           nu_dif, nu_dif_m, drr, drrm, drr2, wfm, dth,
                           margin, ffr, ffth):
    """子午面運動量に働く粘性力を加算する.

    圧縮性の粘性応力テンソル

    .. math::
       \\tau_{ij} = \\rho_0\\nu\\left(
         \\partial_i v_j + \\partial_j v_i
         - \\tfrac{2}{3}\\delta_{ij}\\nabla\\cdot\\boldsymbol{v}\\right)

    の :math:`r`, :math:`\\theta` 成分を球座標で展開したもの. 齋藤 (2024)
    ``calculation.f90`` の ``reyno`` を移植した.

    発散として書ける部分は面フラックス配列 (``ffr``, ``ffth``) に置き,
    幾何因子から生じる残りをセル中心の源項として加える. 面フラックスは
    境界でリテラル 0.0 に落とすので, 粘性を通じて境界から運動量が
    出入りすることはない.

    子午面運動量は保存量ではないので厳密な telescoping は要求されないが,
    フラックスを 1 回だけ計算して 2 セルで共有する規律は角運動量と揃えて
    おく (人工粘性を足したときに同じ枠組みで扱えるようにするため).
    """
    ixg, jxg = dq_mr.shape
    idth = 1.0/dth
    c43 = 4.0/3.0
    c23 = 2.0/3.0

    # =====================================================================
    # r 方向運動量
    # =====================================================================
    # --- r 面フラックス ---------------------------------------------------
    for i in prange(margin, ixg - margin + 1):
        w = wfm[i]
        w1 = 1.0 - w
        idrrm = 1.0/drrm[i]          # 面での勾配 = セル中心間距離で割る
        rop3 = w*ro0[i - 1]*rr[i - 1]**3 + w1*ro0[i]*rr[i]**3
        rop1a = ro0[i]*rr[i]
        rop1b = ro0[i - 1]*rr[i - 1]
        for j in range(margin, jxg - margin):
            dvr = (vrr[i, j]/rr[i] - vrr[i - 1, j]/rr[i - 1])*idrrm
            div_a = (sinTH[i, j + 1]*vth[i, j + 1]
                     - sinTH[i, j - 1]*vth[i, j - 1])*0.5*idth
            div_b = (sinTH[i - 1, j + 1]*vth[i - 1, j + 1]
                     - sinTH[i - 1, j - 1]*vth[i - 1, j - 1])*0.5*idth
            ffr[i, j] = -nu_dif_m[i]*(c43*rop3*sinTH[i, j]*dvr
                             - c23*0.5*(rop1a*div_a + rop1b*div_b))
    zero_boundary_faces_r(ffr, margin)

    # --- theta 面フラックス ----------------------------------------------
    for i in prange(margin, ixg - margin):
        r2 = rr[i]*rr[i]
        idrr2 = 1.0/drr2[i]
        for j in range(margin, jxg - margin + 1):
            sh_a = sinTH[i, j]*(vth[i + 1, j]/rr[i + 1]
                                - vth[i - 1, j]/rr[i - 1])*idrr2
            sh_b = sinTH[i, j - 1]*(vth[i + 1, j - 1]/rr[i + 1]
                                    - vth[i - 1, j - 1]/rr[i - 1])*idrr2
            ffth[i, j] = -nu_dif[i]*ro0[i]*(
                r2*0.5*(sh_a + sh_b)
                + 0.5*(sinTH[i, j] + sinTH[i, j - 1])*(vrr[i, j] - vrr[i, j - 1])*idth)
    zero_boundary_faces_th(ffth, margin)

    add_flux_divergence(dq_mr, ffr, ffth, drr, dth, margin)

    # --- 幾何因子による源項 ----------------------------------------------
    for i in prange(margin, ixg - margin):
        r2 = rr[i]*rr[i]
        idrr2 = 1.0/drr2[i]
        for j in range(margin, jxg - margin):
            dvr = (vrr[i + 1, j]/rr[i + 1] - vrr[i - 1, j]/rr[i - 1])*idrr2
            dsv = (sinTH[i, j + 1]*vth[i, j + 1]
                   - sinTH[i, j - 1]*vth[i, j - 1])*0.5*idth
            dq_mr[i, j] += -nu_dif[i]*ro0[i]*(-c43*r2*sinTH[i, j]*dvr + c23*dsv)

    # =====================================================================
    # theta 方向運動量
    # =====================================================================
    for i in prange(margin, ixg - margin + 1):
        w = wfm[i]
        w1 = 1.0 - w
        idrrm = 1.0/drrm[i]
        rop3 = w*ro0[i - 1]*rr[i - 1]**3 + w1*ro0[i]*rr[i]**3
        rop1a = ro0[i]*rr[i]
        rop1b = ro0[i - 1]*rr[i - 1]
        for j in range(margin, jxg - margin):
            dvt = (vth[i, j]/rr[i] - vth[i - 1, j]/rr[i - 1])*idrrm
            dva = (vrr[i, j + 1] - vrr[i, j - 1])*0.5*idth
            dvb = (vrr[i - 1, j + 1] - vrr[i - 1, j - 1])*0.5*idth
            ffr[i, j] = -nu_dif_m[i]*sinTH[i, j]*(rop3*dvt
                                         + 0.5*(rop1a*dva + rop1b*dvb))
    zero_boundary_faces_r(ffr, margin)

    for i in prange(margin, ixg - margin):
        r2 = rr[i]*rr[i]
        idrr2 = 1.0/drr2[i]
        for j in range(margin, jxg - margin + 1):
            sh_a = sinTH[i, j]*(vrr[i + 1, j]/rr[i + 1]
                                - vrr[i - 1, j]/rr[i - 1])*idrr2
            sh_b = sinTH[i, j - 1]*(vrr[i + 1, j - 1]/rr[i + 1]
                                    - vrr[i - 1, j - 1]/rr[i - 1])*idrr2
            ffth[i, j] = -nu_dif[i]*ro0[i]*(
                -c23*r2*0.5*(sh_a + sh_b)
                + c43*0.5*(sinTH[i, j] + sinTH[i, j - 1])*(vth[i, j] - vth[i, j - 1])*idth
                - c23*0.5*(cosTH[i, j]*vth[i, j] + cosTH[i, j - 1]*vth[i, j - 1]))
    zero_boundary_faces_th(ffth, margin)

    add_flux_divergence(dq_mt, ffr, ffth, drr, dth, margin)

    for i in prange(margin, ixg - margin):
        r2 = rr[i]*rr[i]
        idrr2 = 1.0/drr2[i]
        for j in range(margin, jxg - margin):
            s = sinTH[i, j]
            c = cosTH[i, j]
            dvt_r = (vth[i + 1, j]/rr[i + 1] - vth[i - 1, j]/rr[i - 1])*idrr2
            dvr_r = (vrr[i + 1, j]/rr[i + 1] - vrr[i - 1, j]/rr[i - 1])*idrr2
            dvr_t = (vrr[i, j + 1] - vrr[i, j - 1])*0.5*idth
            dvt_t = (vth[i, j + 1] - vth[i, j - 1])*0.5*idth
            dq_mt[i, j] += nu_dif[i]*ro0[i]*(
                r2*s*dvt_r + s*dvr_t
                + c23*r2*c*dvr_r + c23*c*dvt_t
                - c43*c*c/s*vth[i, j])


# ---------------------------------------------------------------------------
# エントロピー
# ---------------------------------------------------------------------------
@kernel()
def entropy_rhs(dse1, se1, vrr, vth, ro0, tm0, pr0, hp, delta, kappa, kappa_m,
                JM, iJM, rr, sinTH, sinTHm, gamma, drr, drrm, drr2, wfm,
                dth, margin, ffr, ffth):
    """エントロピー方程式の右辺を ``dse1`` に加算する (Rempel 2006 式 5).

    .. math::
       \\frac{\\partial s_1}{\\partial t}
         = -v_r\\frac{\\partial s_1}{\\partial r}
           -\\frac{v_\\theta}{r}\\frac{\\partial s_1}{\\partial\\theta}
           + v_r\\frac{\\gamma\\delta}{H_p}
           + \\frac{1}{\\rho_0 T_0}
             \\nabla\\cdot\\!\\left(\\kappa_t\\rho_0 T_0\\nabla s_1\\right)

    加熱項 (:math:`Q` とオーム散逸) は :func:`add_dissipative_heating` で
    別に加える.

    :math:`s_1` の規格化
    ---------------------
    Rempel は :math:`s=\\ln(p\\rho^{-\\gamma})` を :math:`c_v` で割った
    **無次元エントロピー**を使う (:math:`c_p` ではない). 状態方程式は

    .. math:: p_1 = p_0\\left(\\gamma\\frac{\\rho_1}{\\rho_0} + s_1\\right)

    で, 齋藤 (2024) コードの ``en1`` と同一の量である.

    背景エントロピー勾配
    --------------------
    Rempel 2005 式 (8) より :math:`ds_0/dr = -\\gamma\\delta/H_p` なので,
    移流項 :math:`-v_r\\,ds_0/dr` は :math:`+v_r\\gamma\\delta/H_p` になる.
    :math:`\\delta=\\nabla-\\nabla_{\\rm ad}` は超断熱度で, 正が対流不安定.

    形式について
    ------------
    移流は Rempel に合わせて**移流形** (保存形ではない) で書く. 熱伝導だけは
    面フラックス配列を使い, 境界面をゼロにすることで境界条件
    :math:`\\partial s_1/\\partial r=0` を厳密に実装する.
    エントロピーは保存量ではない (散逸で生成される) ので, 移流形でよい.
    """
    ixg, jxg = dse1.shape
    idth = 1.0/dth

    # --- 熱伝導フラックス (面で1回だけ計算) ------------------------------
    for i in prange(margin, ixg - margin + 1):
        # 面上の kappa*rho0*T0. 隣接2セルで同じ値を使う
        w = wfm[i]
        w1 = 1.0 - w
        idrrm = 1.0/drrm[i]          # 面での勾配
        c = kappa_m[i]*(w*ro0[i - 1]*tm0[i - 1] + w1*ro0[i]*tm0[i])
        rmf = w*rr[i - 1] + w1*rr[i]
        rm2 = rmf*rmf
        for j in range(jxg):
            ffr[i, j] = -c*rm2*sinTH[i, j]*(se1[i, j] - se1[i - 1, j])*idrrm
    zero_boundary_faces_r(ffr, margin)

    for i in prange(ixg):
        c = kappa[i]*ro0[i]*tm0[i]
        for j in range(margin, jxg - margin + 1):
            ffth[i, j] = -c*sinTHm[i, j]*(se1[i, j] - se1[i, j - 1])*idth
    zero_boundary_faces_th(ffth, margin)

    # --- 移流 + 背景勾配 + 熱伝導 ----------------------------------------
    for i in prange(margin, ixg - margin):
        idrr = 1.0/drr[i]            # 発散はセル幅
        idrr2 = 1.0/drr2[i]          # 移流の中心差分は 2 セル幅
        for j in range(margin, jxg - margin):
            dsdr = (se1[i + 1, j] - se1[i - 1, j])*idrr2
            dsdt = (se1[i, j + 1] - se1[i, j - 1])*0.5*idth
            cond = -((ffr[i + 1, j] - ffr[i, j])*idrr
                     + (ffth[i, j + 1] - ffth[i, j])*idth)
            dse1[i, j] += (-vrr[i, j]*dsdr - vth[i, j]*dsdt/rr[i]
                           + vrr[i, j]*gamma*delta[i]/hp[i]
                           + cond*iJM[i, j]/(ro0[i]*tm0[i]))


@kernel()
def add_dissipative_heating(dse1, heat, dq_mr, dq_mt, vrr, vth,
                            pr0, iJM, gamma, margin):
    """散逸で失われたエネルギーをエントロピーに戻す.

    ``heat`` には**局所的な**エネルギー変換率 :math:`-F\\cdot\\nabla u` を
    :func:`S2MFD.physics.conservative.add_flux_work` で積んでおくこと.
    :math:`u\\,\\partial q/\\partial t` をそのまま使うと, 体積積分は
    合っていても局所的には発散の分だけずれ, 角運動量の場合は
    :math:`\\Omega_0/\\Omega_1` 倍 (20-100 倍) の偽の加熱・冷却になる.

    子午面運動量については源項 (幾何因子由来) も仕事をするので,
    フラックス由来の分と合わせて ``dq_mr``, ``dq_mt`` から
    :math:`v\\cdot\\dot q_m` として評価する. こちらは :math:`v^2` の
    オーダーで大きな定数を含まないため, 発散分のずれも同じオーダーに
    留まる.

    無次元エントロピー (:math:`c_v` で規格化) では
    :math:`\\rho_0T_0c_v = p_0/(\\gamma-1)` なので

    .. math:: \\frac{\\partial s_1}{\\partial t} \\mathrel{+}=
              \\frac{\\gamma-1}{p_0}\\,Q

    となる (Rempel 2006 式 5 の第 4 項と同じ係数). 係数はちょうど 1 で,
    「運動量方程式が奪った分をそのまま渡す」ことになる — ユーザー要求
    「人工粘性によって散逸した運動エネルギー・磁場エネルギーをエントロピーの
    式に足す」を離散レベルで満たす形である.

    Λ 効果の寄与は**負の加熱 (冷却)** になる. 差動回転にエネルギーを渡す側
    だからで, Rempel 2005 の記述「Q は粘性散逸による加熱項と Λ 効果による
    冷却項を含み, 後者が一般に支配的」がそのまま再現される.
    """
    ixg, jxg = dse1.shape
    for i in prange(margin, ixg - margin):
        c = (gamma - 1.0)/pr0[i]
        for j in range(margin, jxg - margin):
            q = heat[i, j] - (vrr[i, j]*dq_mr[i, j] + vth[i, j]*dq_mt[i, j])
            dse1[i, j] += c*q*iJM[i, j]


@kernel()
def add_ohmic_heating(dse1, brr, bth, bph, eta, pr0, RR, sinTH,
                      gamma, drr2, dth, margin):
    """オーム散逸をエントロピーに加える (Rempel 2006 式 5 の最終項).

    .. math::
       \\frac{\\partial s_1}{\\partial t}\\mathrel{+}=
         \\frac{\\gamma-1}{p_0}\\,\\frac{\\eta_t|\\nabla\\times B|^2}{4\\pi}

    原論文の式 (5) と式 (33) には :math:`\\mu_0` が印字されていないが,
    次元が合わないため補っている (CGS なので :math:`1/4\\pi`).
    式 (25) の :math:`E_B=\\int B_\\Phi^2/(2\\mu_0)\\,dV` には
    :math:`\\mu_0` があることから, 印刷上の脱落と判断した.
    詳細は ``doc/dev_records/2026-08-19_rempel_equations.md`` §11.
    """
    ixg, jxg = dse1.shape
    idth = 1.0/dth
    for i in prange(margin, ixg - margin):
        idrr2 = 1.0/drr2[i]
        c = (gamma - 1.0)/pr0[i]/FOUR_PI
        for j in range(margin, jxg - margin):
            r = RR[i, j]
            s = sinTH[i, j]
            jr = (sinTH[i, j + 1]*bph[i, j + 1]
                  - sinTH[i, j - 1]*bph[i, j - 1])*0.5*idth/(r*s)
            jt = -(RR[i + 1, j]*bph[i + 1, j]
                   - RR[i - 1, j]*bph[i - 1, j])*idrr2/r
            jp = ((RR[i + 1, j]*bth[i + 1, j]
                   - RR[i - 1, j]*bth[i - 1, j])*idrr2
                  - (brr[i, j + 1] - brr[i, j - 1])*0.5*idth)/r
            dse1[i, j] += c*eta[i, j]*(jr*jr + jt*jt + jp*jp)


# ---------------------------------------------------------------------------
# 保存量 <-> プリミティブ変数
# ---------------------------------------------------------------------------
@kernel()
def to_primitive_om1(q_om, iJL, om1, margin):
    """:math:`q_L \\to \\Omega_1`. 物理セルのみ変換する (ゴーストは境界条件で埋める)."""
    ixg, jxg = q_om.shape
    for i in prange(margin, ixg - margin):
        for j in range(margin, jxg - margin):
            om1[i, j] = q_om[i, j]*iJL[i, j]


@kernel()
def to_primitive_ro1(q_ro, iJM, ro1, margin):
    """:math:`q_\\rho \\to \\rho_1`. 物理セルのみ変換する."""
    ixg, jxg = q_ro.shape
    for i in prange(margin, ixg - margin):
        for j in range(margin, jxg - margin):
            ro1[i, j] = q_ro[i, j]*iJM[i, j]
