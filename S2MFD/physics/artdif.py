"""人工拡散 (slope-limited diffusion) — 保存形版.

Rempel (2005, 2006) は MacCormack の交互風上/風下差分を使っており, その
スキームに内在する数値散逸で安定性を得ている. 人工粘性を明示的には
入れていない (両論文とも記述なし).

S2MFD は SSP-RK2 + 中心差分で**数値散逸をまったく持たない**ため, 別途
人工拡散が必要になる. これは論文からの逸脱ではなく, 異なる時間積分法を
使うことの必然的な帰結である.

方式は Rempel (2014, ApJ 789, 132) §2.1 の slope-limited diffusion (SLD).
実装は R2D2 (Hotta) の ``src/include/artdif_func.F95`` に合わせてある.

.. math::
   u_l &= u_i + \\tfrac12\\Delta u_i, \\qquad
   u_r = u_{i+1} - \\tfrac12\\Delta u_{i+1} \\\\
   \\Delta u_i &= {\\rm minmod}_\\epsilon\\!\\left[
      \\tfrac12(u_{i+1}-u_{i-1}),\\,
      \\epsilon(u_{i+1}-u_i),\\,
      \\epsilon(u_i-u_{i-1})\\right] \\\\
   f_{i+1/2} &= -\\tfrac12 c_{i+1/2}\\,\\Phi_h\\,(u_r-u_l) \\\\
   \\Phi_h &= \\max\\!\\left[0,\\,
      1 + h\\left(\\min(1, r) - 1\\right)\\right],\\quad
      r = \\frac{u_r-u_l}{u_{i+1}-u_i}

:math:`r \\le 0` (反拡散になる向き) では :math:`\\Phi_h=0`.

なぜ「解像された場では効かない」のか
------------------------------------
滑らかで局所的に線形な場では, 左右からの再構成が面上で一致するので
:math:`u_r-u_l=0` となりフラックスが厳密に消える. 逆に格子スケールの
振動ではリミタが傾きをゼロにするので :math:`u_r-u_l=u_{i+1}-u_i`,
:math:`r=1`, :math:`\\Phi_h=1` となり最大拡散
:math:`\\tfrac12 c\\Delta x` が効く.

:math:`h>1` にすると :math:`r<1-1/h` の領域でフラックスが**完全に切れる**.
Rempel は :math:`h=2` を使う (R2D2 も同じ) ので, :math:`r<0.5` では
人工拡散がゼロになる. :math:`h=0` は 2 次 Lax-Friedrichs に一致する.

パラメタ (R2D2 の値)
--------------------
* :math:`h` = ``fh`` = 2.0
* :math:`\\epsilon` = ``ep`` = 2.0 (generalized minmod. 2 で MC リミタ)
* 特性速度 :math:`c = |v| + v_A + 0.3\\,c_{s,\\rm eff}`

音速に 0.3 を掛けるのは, 抑制後とはいえ音速が流れより 2 桁速く, そのままだと
人工拡散がモデルの依存する低拡散領域 (オーバーシュート層の
:math:`\\kappa_t`, 放射層の :math:`\\nu_{\\rm dif}`) を上回ってしまうため.
CFL には抑制なしの :math:`|v|+v_A+c_{s,\\rm eff}` を使う.

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

from ._jit import kernel, prange

from S2MFD.physics.conservative import (
    zero_boundary_faces_r, zero_boundary_faces_th,
    add_flux_divergence, add_flux_divergence_scaled, add_flux_work,
)

__all__ = ['sld_flux_r', 'sld_flux_th', 'sld_diffuse_meridional',
           'sld_diffuse', 'sld_diffuse_work', 'sld_diffuse_scaled',
           'sld_diffuse_primitive', 'sld_diffusivity_max',
           'hyper_flux_r', 'hyper_diffuse_r', 'hyper_diffuse_r_primitive',
           'mean_profile_diffuse_r', 'mean_profile_diffuse_r_primitive']


@njit(inline='always')
def _minmod3(d_dwn, d_upp, ep):
    """一般化 minmod リミタ (Rempel 2014 式 8, R2D2 の ``artdif_dqq``).

    3 つの候補 :math:`\\epsilon\\Delta_{\\rm dwn}`,
    :math:`\\epsilon\\Delta_{\\rm upp}`,
    :math:`\\tfrac12(\\Delta_{\\rm dwn}+\\Delta_{\\rm upp})` のうち,
    すべて同符号ならゼロに最も近いものを, 符号が揃わなければ 0 を返す.
    :math:`\\epsilon=2` で monotonized central (MC) リミタになる.
    """
    d_cen = 0.5*(d_dwn + d_upp)
    a, b, c = ep*d_dwn, ep*d_upp, d_cen
    dmax = a if a > b else b
    if c > dmax:
        dmax = c
    dmin = a if a < b else b
    if c < dmin:
        dmin = c
    return (dmax if dmax < 0.0 else 0.0) + (dmin if dmin > 0.0 else 0.0)


@njit(inline='always')
def _sld_face_flux(q_dwn, q_upp, dq_dwn, dq_upp, fh, cc, jac):
    """面フラックス (Rempel 2014 式 9-10, R2D2 の ``artdif_flux``).

    ``jac`` は保存量のヤコビアン (保存形に載せるための係数).
    """
    ql = q_dwn + 0.5*dq_dwn
    qr = q_upp - 0.5*dq_upp
    d = q_upp - q_dwn
    # ゼロ割り回避 (R2D2 と同じく符号を保ったまま下限を張る)
    ad = abs(d)
    d = (1.0 if d >= 0.0 else -1.0)*(ad if ad > 1.0e-20 else 1.0e-20)
    ra = (qr - ql)/d
    if ra > 1.0:
        ra = 1.0
    if ra <= 0.0:
        return 0.0            # 反拡散にはしない
    pp = 1.0 + fh*(ra - 1.0)
    if pp < 0.0:
        pp = 0.0
    return -0.5*cc*jac*pp*(qr - ql)


@kernel()
def sld_flux_r(uu, jac_face, cspeed, fh, ep, drr, drrm, margin, out):
    """r 方向の SLD フラックス. 面 ``i`` はセル ``i-1`` と ``i`` の境界.

    非一様格子では**単位長さあたりの傾き**で再構成する。生の差分のままだと
    線形な場でもセル幅の違いだけで minmod がリミタを発動し、**滑らかな場に
    人工拡散が乗ってしまう**。一様格子では ``drr == drrm`` なので
    従来の式と代数的に同一。
    """
    ixg, jxg = uu.shape
    for j in prange(jxg):
        for i in range(margin, ixg - margin + 1):
            # セル i-1 と i の再構成傾き
            im2 = i - 2 if i - 2 >= 0 else i - 1
            ip1 = i + 1 if i + 1 < ixg else i
            gm2 = (uu[i - 1, j] - uu[im2, j])/drrm[i - 1]
            gm1 = (uu[i, j] - uu[i - 1, j])/drrm[i]
            gp1 = (uu[ip1, j] - uu[i, j])/drrm[ip1]
            dq_dwn = drr[i - 1]*_minmod3(gm2, gm1, ep)
            dq_upp = drr[i]*_minmod3(gm1, gp1, ep)
            out[i, j] = _sld_face_flux(uu[i - 1, j], uu[i, j], dq_dwn, dq_upp,
                                       fh, cspeed[i, j], jac_face[i, j])


@kernel()
def sld_flux_th(uu, jac_face, cspeed, fh, ep, margin, out):
    """theta 方向の SLD フラックス. 面 ``j`` はセル ``j-1`` と ``j`` の境界."""
    ixg, jxg = uu.shape
    for i in prange(ixg):
        for j in range(margin, jxg - margin + 1):
            jm2 = j - 2 if j - 2 >= 0 else j - 1
            jp1 = j + 1 if j + 1 < jxg else j
            dq_dwn = _minmod3(uu[i, j - 1] - uu[i, jm2],
                              uu[i, j] - uu[i, j - 1], ep)
            dq_upp = _minmod3(uu[i, j] - uu[i, j - 1],
                              uu[i, jp1] - uu[i, j], ep)
            out[i, j] = _sld_face_flux(uu[i, j - 1], uu[i, j], dq_dwn, dq_upp,
                                       fh, cspeed[i, j], jac_face[i, j])


@njit(fastmath=False)
def sld_diffuse(dqq, uu, jac_r, jac_th, csp_r, csp_th, fh, ep,
                drr, drrm, dth, margin, ffr, ffth):
    """プリミティブ変数 ``uu`` に人工拡散を掛け, 保存量の時間微分に加算する.

    境界面のフラックスはリテラル 0.0 に落とすので, 人工拡散を通じて
    保存量が境界から出入りすることはない (ユーザー要求
    「境界から保存量が出ていかないように。人工粘性も物理も」).
    """
    sld_flux_r(uu, jac_r, csp_r, fh, ep, drr, drrm, margin, ffr)
    zero_boundary_faces_r(ffr, margin)
    sld_flux_th(uu, jac_th, csp_th, fh, ep, margin, ffth)
    zero_boundary_faces_th(ffth, margin)
    add_flux_divergence(dqq, ffr, ffth, drr, dth, margin)


@njit(fastmath=False)
def sld_diffuse_work(dqq, heat, uu, jac_r, jac_th, csp_r, csp_th, fh, ep,
                     drr, drrm, dth, margin, ffr, ffth):
    """:func:`sld_diffuse` に加えて, 局所的な散逸率を ``heat`` に積む.

    角運動量に掛けるときはこちらを使う. :math:`\\Omega_0` は
    :math:`-F\\cdot\\nabla\\Omega` の微分で消えるので, 局所的な加熱が
    :math:`\\Omega_0/\\Omega_1` 倍に化ける問題を避けられる
    (:func:`S2MFD.physics.conservative.add_flux_work` の説明を参照).
    """
    sld_flux_r(uu, jac_r, csp_r, fh, ep, drr, drrm, margin, ffr)
    zero_boundary_faces_r(ffr, margin)
    sld_flux_th(uu, jac_th, csp_th, fh, ep, margin, ffth)
    zero_boundary_faces_th(ffth, margin)
    add_flux_divergence(dqq, ffr, ffth, drr, dth, margin)
    add_flux_work(heat, ffr, ffth, uu, drr, dth, margin)


@njit(fastmath=False)
def sld_diffuse_scaled(dqq, uu, jac_r, jac_th, csp_r, csp_th, fh, ep,
                       drr, drrm, dth, margin, scale, ffr, ffth):
    """発散に動径依存の係数が掛かる版 (密度に使う).

    音速抑制法では保存量が :math:`\\int\\xi_s^2\\rho_1\\,dV` なので,
    連続の式と同じく :math:`\\xi_s^{-2}` を発散全体に掛けないと質量保存が
    壊れる (:math:`\\xi_s` が動径に依存する場合).
    """
    sld_flux_r(uu, jac_r, csp_r, fh, ep, drr, drrm, margin, ffr)
    zero_boundary_faces_r(ffr, margin)
    sld_flux_th(uu, jac_th, csp_th, fh, ep, margin, ffth)
    zero_boundary_faces_th(ffth, margin)
    add_flux_divergence_scaled(dqq, ffr, ffth, drr, dth, margin, scale)


@kernel()
def sld_diffuse_primitive(duu, uu, jac_r, jac_th, ijac, csp_r, csp_th, fh, ep,
                          drr, drrm, dth, margin, ffr, ffth):
    """保存形ではなく直接解いているプリミティブ変数に人工拡散を掛ける.

    エントロピーのように保存量として持っていない変数に使う.
    フラックスは保存形と同じヤコビアン付きで作り, 最後にセル中心の
    ヤコビアンで割って :math:`\\partial u/\\partial t` に変換する.
    """
    sld_flux_r(uu, jac_r, csp_r, fh, ep, drr, drrm, margin, ffr)
    zero_boundary_faces_r(ffr, margin)
    sld_flux_th(uu, jac_th, csp_th, fh, ep, margin, ffth)
    zero_boundary_faces_th(ffth, margin)
    ixg, jxg = duu.shape
    idth = 1.0/dth
    for i in prange(margin, ixg - margin):
        idrr = 1.0/drr[i]
        for j in range(margin, jxg - margin):
            duu[i, j] -= ((ffr[i + 1, j] - ffr[i, j])*idrr
                          + (ffth[i, j + 1] - ffth[i, j])*idth)*ijac[i, j]


@njit(fastmath=False)
def sld_diffusivity_max(csp_r, csp_th, drr, dth, rr, margin):
    """人工拡散の実効拡散係数の上限 :math:`\\tfrac12 c\\Delta x` を返す (CFL 用).

    :math:`\\Phi_h\\le1` なので, これが実際に到達しうる最大値になる
    (Rempel 2014 §2.1: "the maximum diffusivity of
    :math:`0.5\\,c_{i+1/2}\\,\\Delta x`").
    """
    ixg, jxg = csp_r.shape
    kmax = 0.0
    for i in range(margin, ixg - margin):
        dl = drr[i] if drr[i] < rr[i]*dth else rr[i]*dth
        for j in range(margin, jxg - margin):
            c = csp_r[i, j] if csp_r[i, j] > csp_th[i, j] else csp_th[i, j]
            k = 0.5*c*dl
            if k > kmax:
                kmax = k
    return kmax


@kernel()
def sld_diffuse_meridional(dq_mr, dq_mt, vrr, vth, jac_r, jac_th,
                           csp_r, csp_th, fh, ep, drr, drrm, dth, margin,
                           ffr1, ffth1, ffr2, ffth2):
    """子午面速度 :math:`(v_r, v_\\theta)` に人工拡散を掛ける (幾何項つき).

    球座標の基底ベクトルは :math:`\\theta` に依存する
    (:math:`\\partial\\hat e_r/\\partial\\theta=\\hat e_\\theta`,
    :math:`\\partial\\hat e_\\theta/\\partial\\theta=-\\hat e_r`) ので,
    :math:`v_r` と :math:`v_\\theta` をスカラーとして独立に拡散させると
    **間違いになる**. :math:`\\theta` 方向の掃引では

    .. math::
       \\frac{\\partial}{\\partial\\theta}(F_r\\hat e_r + F_\\theta\\hat e_\\theta)
       = (\\partial_\\theta F_r - F_\\theta)\\hat e_r
       + (\\partial_\\theta F_\\theta + F_r)\\hat e_\\theta

    となるので, 成分が混ざる項を足す必要がある. 保存量にヤコビアンを
    吸収した本実装では, 面フラックス配列をそのままセル中心へ平均して

    .. math::
       \\dot q_{m,r} \\mathrel{+}= \\langle F^\\theta_{v_\\theta}\\rangle,
       \\qquad
       \\dot q_{m,\\theta} \\mathrel{-}= \\langle F^\\theta_{v_r}\\rangle

    と書ける (R2D2 の ``artdif_spherical.F90`` の ``j1==1`` ブロックと同じ).

    :math:`r` 方向の掃引には幾何項は要らない (基底は :math:`r` に依らない).
    :math:`\\Omega_1`, :math:`\\rho_1`, :math:`s_1` はスカラーなので
    そもそも不要.
    """
    # --- v_r ---------------------------------------------------------------
    sld_flux_r(vrr, jac_r, csp_r, fh, ep, drr, drrm, margin, ffr1)
    zero_boundary_faces_r(ffr1, margin)
    sld_flux_th(vrr, jac_th, csp_th, fh, ep, margin, ffth1)
    zero_boundary_faces_th(ffth1, margin)
    add_flux_divergence(dq_mr, ffr1, ffth1, drr, dth, margin)

    # --- v_theta -----------------------------------------------------------
    sld_flux_r(vth, jac_r, csp_r, fh, ep, drr, drrm, margin, ffr2)
    zero_boundary_faces_r(ffr2, margin)
    sld_flux_th(vth, jac_th, csp_th, fh, ep, margin, ffth2)
    zero_boundary_faces_th(ffth2, margin)
    add_flux_divergence(dq_mt, ffr2, ffth2, drr, dth, margin)

    # --- 幾何項 (theta 掃引での基底の回転) ---------------------------------
    ixg, jxg = dq_mr.shape
    for i in prange(margin, ixg - margin):
        for j in range(margin, jxg - margin):
            dq_mr[i, j] += 0.5*(ffth2[i, j] + ffth2[i, j + 1])
            dq_mt[i, j] -= 0.5*(ffth1[i, j] + ffth1[i, j + 1])


# ---------------------------------------------------------------------------
# 4 次のハイパー拡散 (Rempel 2014 §2.1 の "fourth hyper-diffusion term")
# ---------------------------------------------------------------------------
@kernel()
def hyper_flux_r(uu, jac_face, vadv, h4, margin, out):
    """r 方向の 4 次ハイパー拡散フラックス.

    Rempel (2014) §2.1:

        "We also added an additional optional fourth hyper-diffusion term
         that scales with the advection velocity and acts only in the vertical
         direction on the quantities log(rho), v_z, and eps. This term allows
         us to damp some low level spurious oscillations on the grid scale
         that are too small to cause monotonicity changes in the presence of
         a background gradient (stratification) and go mostly undetected by
         the slope-limited diffusion scheme."

    なぜ SLD では取れないか
    -----------------------
    背景勾配 (成層や大規模構造) があると, MC リミタは「単調で解像されている」
    と判断してしまう. その上に乗った微小な格子スケールの振動は単調性を
    壊さないので, :math:`u_r-u_l\\simeq0` のまま
    :math:`\\Phi_h=0` になり, SLD が素通りさせる.

    4 階微分なら**線形・2 次の背景を完全に見ない**ので, 背景勾配に隠れた
    格子スケール成分だけを選択的に潰せる.

    離散化
    ------
    :math:`\\partial u/\\partial t = -\\nu_4\\partial^4u/\\partial x^4`
    をフラックス形 :math:`F=\\nu_4\\partial^3u/\\partial x^3` で書き,
    :math:`\\nu_4 = h_4|v|\\Delta x^3` と取ると

    .. math::
       F_{i-1/2} = h_4\\,|v|_{i-1/2}\\,J_{i-1/2}\\,
                   (u_{i+1} - 3u_i + 3u_{i-1} - u_{i-2})

    となり :math:`\\Delta x` が消える. 1 セルおきに符号が変わる振動に対して
    3 階差分は :math:`-8(-1)^i` と最大になり, 線形・2 次の場では厳密にゼロ.

    ``vadv`` は面上の移流速度 :math:`|\\boldsymbol v|`
    (音速ではない — 背景勾配に隠れた振動は流れに乗って運ばれるため).
    """
    ixg, jxg = uu.shape
    for j in prange(jxg):
        for i in range(margin, ixg - margin + 1):
            ip1 = i + 1 if i + 1 < ixg else i
            im2 = i - 2 if i - 2 >= 0 else i - 1
            d3 = uu[ip1, j] - 3.0*uu[i, j] + 3.0*uu[i - 1, j] - uu[im2, j]
            out[i, j] = h4*vadv[i, j]*jac_face[i, j]*d3


@kernel()
def hyper_diffuse_r(dqq, uu, jac_r, vadv_r, h4, drr, dth, margin, ffr, ffth):
    """保存量に 4 次ハイパー拡散を加える (r 方向のみ)."""
    hyper_flux_r(uu, jac_r, vadv_r, h4, margin, ffr)
    zero_boundary_faces_r(ffr, margin)
    ixg, jxg = dqq.shape
    for i in prange(ixg):
        for j in range(jxg):
            ffth[i, j] = 0.0
    add_flux_divergence(dqq, ffr, ffth, drr, dth, margin)


@kernel()
def hyper_diffuse_r_primitive(duu, uu, jac_r, ijac, vadv_r, h4, drr, margin, ffr):
    """保存形で持っていないプリミティブ変数に 4 次ハイパー拡散を加える."""
    hyper_flux_r(uu, jac_r, vadv_r, h4, margin, ffr)
    zero_boundary_faces_r(ffr, margin)
    ixg, jxg = duu.shape
    for i in prange(margin, ixg - margin):
        idrr = 1.0/drr[i]
        for j in range(margin, jxg - margin):
            duu[i, j] -= (ffr[i + 1, j] - ffr[i, j])*idrr*ijac[i, j]


# ---------------------------------------------------------------------------
# 緯度平均プロファイルへの拡散
# ---------------------------------------------------------------------------
@njit(fastmath=False)
def _mean_profile(uu, jac, margin, wsum, prof):
    """質量重み付きの theta 平均プロファイルを作る."""
    ixg, jxg = uu.shape
    for i in range(margin, ixg - margin):
        w = 0.0
        a = 0.0
        for j in range(margin, jxg - margin):
            w += jac[i, j]
            a += jac[i, j]*uu[i, j]
        wsum[i] = w
        prof[i] = a/w if w != 0.0 else 0.0


@njit(fastmath=False)
def mean_profile_diffuse_r(dqq, uu, jac, kappa, drr, drrm, margin,
                           wsum, prof, dq1):
    """**theta 方向に平均した動径プロファイルにだけ**拡散をかける.

    なぜこれが要るか
    ----------------
    格子スケールの振動のうち, **緯度平均して初めて見える成分**がある.
    各点では滑らかな場の上に乗った数 % の揺らぎに過ぎないので, SLD の
    リミタは「解像されている」と判断して素通りさせる (実測では
    :math:`r\\simeq0.5` で :math:`h=2` の切断閾値ちょうど). しかし
    緯度平均すると滑らかな成分が打ち消し合い (質量保存により
    :math:`\\int v_r\\sin\\theta\\,d\\theta\\simeq0`), 振動だけが残る.

    この成分は浮力と圧力勾配によって格子スケールで駆動されており
    (collocated 格子の odd-even 分離), SLD と物理粘性との釣り合いで
    有限振幅の**強制平衡**に落ち着いてしまう.

    なぜ弱くてよいか
    ----------------
    :math:`v_r` の緯度平均は質量保存によりほぼゼロなので, そこを
    滑らかにしても物理的な解を削らない. 標的が非常に限定されているので
    係数は小さくてよく, 演算も 1 次元 + ブロードキャストで安い.

    保存
    ----
    質量重み付き平均を取り, 動径フラックスの発散を同じ重みで各緯度へ
    配分するので, :math:`\\sum_j` が厳密に元の :math:`dQ_i` に戻り,
    :math:`\\sum_i` は telescoping する. 境界面はリテラル 0.0.
    各緯度への配分が ``jac`` に比例するため, プリミティブ量への寄与は
    **緯度によらず一定** — つまり theta 平均成分だけを変える.

    Parameters
    ----------
    kappa : float
        拡散係数 [cm^2/s].
    wsum, prof, dq1 : numpy.ndarray
        1 次元の作業配列 (``ixg``,).
    """
    ixg, jxg = dqq.shape
    _mean_profile(uu, jac, margin, wsum, prof)
    # 面フラックス (境界面はゼロのまま) -> セルの変化率
    for i in range(margin, ixg - margin):
        dq1[i] = 0.0
    for i in range(margin, ixg - margin):
        fl = 0.0
        fu = 0.0
        if i > margin:
            fl = (-kappa*0.5*(wsum[i] + wsum[i - 1])
                  * (prof[i] - prof[i - 1])/drrm[i])
        if i < ixg - margin - 1:
            fu = (-kappa*0.5*(wsum[i + 1] + wsum[i])
                  * (prof[i + 1] - prof[i])/drrm[i + 1])
        dq1[i] = -(fu - fl)/drr[i]
    # 質量重みで各緯度へ配分 (プリミティブ量では theta によらず一定の変化)
    for i in range(margin, ixg - margin):
        w = wsum[i]
        if w == 0.0:
            continue
        c = dq1[i]/w
        for j in range(margin, jxg - margin):
            dqq[i, j] += c*jac[i, j]


@njit(fastmath=False)
def mean_profile_diffuse_r_primitive(duu, uu, jac, ijac, kappa, drr, drrm,
                                     margin,
                                     wsum, prof, dq1):
    """保存形で持っていないプリミティブ変数版 (エントロピーなど)."""
    ixg, jxg = duu.shape
    _mean_profile(uu, jac, margin, wsum, prof)
    for i in range(margin, ixg - margin):
        fl = 0.0
        fu = 0.0
        if i > margin:
            fl = (-kappa*0.5*(wsum[i] + wsum[i - 1])
                  * (prof[i] - prof[i - 1])/drrm[i])
        if i < ixg - margin - 1:
            fu = (-kappa*0.5*(wsum[i + 1] + wsum[i])
                  * (prof[i + 1] - prof[i])/drrm[i + 1])
        dq1[i] = -(fu - fl)/drr[i]
    for i in range(margin, ixg - margin):
        w = wsum[i]
        if w == 0.0:
            continue
        c = dq1[i]/w
        for j in range(margin, jxg - margin):
            duu[i, j] += c*jac[i, j]*ijac[i, j]
