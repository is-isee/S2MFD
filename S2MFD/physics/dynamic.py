"""平均場流体ソルバの時間積分ドライバ (Rempel 2006).

:mod:`S2MFD.physics.hydro` の各方程式と :mod:`S2MFD.physics.artdif` の
人工拡散を組み立てて 1 ステップ進める.

状態変数
--------
保存量 (ヤコビアン吸収済み) 4 つとプリミティブ 1 つ:

======  =========================================  ==============
記号    定義                                       プリミティブ
======  =========================================  ==============
q_ro    :math:`r^2\\sin\\theta\\,\\rho_1`            ``ro1``
q_mr    :math:`r^2\\sin\\theta\\,\\rho_0 v_r`        ``vrr``
q_mt    :math:`r^2\\sin\\theta\\,\\rho_0 v_\\theta`   ``vth``
q_om    :math:`r^4\\sin^3\\theta\\,\\rho_0\\Omega_1`  ``om1``
--      エントロピーは移流形で直接解く                ``se1``
======  =========================================  ==============

エントロピーだけ保存形にしないのは Rempel 2006 式 (5) がそう書かれている
ため, かつエントロピーは散逸で生成されるので保存量ではないため.

時間積分は SSP-RK2 (Heun). 保存量の線形結合なので, 各 substep で保存則が
成り立てば RK の合成後も成り立つ.
"""
import warnings

import numpy as np
from numba import njit

from S2MFD.physics import hydro, artdif, stability
from S2MFD.physics.physics_core import poloidal_mag as hydro_poloidal_mag
from S2MFD.physics import conservative as cons

__all__ = ['DynamicSolver', 'apply_radial_bc', 'apply_polar_bc']


# ---------------------------------------------------------------------------
# RK2 の配列演算
# ---------------------------------------------------------------------------
# numpy の ``a + dt*b`` は毎 substep で一時配列を作る。1 step あたり
# 10 本 (保存量 5 つ x 2 substep) になり、108x144 で 1.5 ms/step = 12% を
# 占めていた。演算順序は numpy と同じ (dt*b を作ってから足す) なので
# **結果はビット単位で一致する** (tests で固定)。
@njit(fastmath=False)
def _rk_predictor(out, a, b, dt):
    ixg, jxg = out.shape
    for i in range(ixg):
        for j in range(jxg):
            out[i, j] = a[i, j] + dt*b[i, j]


@njit(fastmath=False)
def _rk_corrector(out, a, b, c, dt):
    ixg, jxg = out.shape
    for i in range(ixg):
        for j in range(jxg):
            out[i, j] = 0.5*(a[i, j] + b[i, j] + dt*c[i, j])


@njit(fastmath=False)
def _mul_into(out, a, b):
    ixg, jxg = out.shape
    for i in range(ixg):
        for j in range(jxg):
            out[i, j] = a[i, j]*b[i, j]


# ---------------------------------------------------------------------------
# 境界条件
# ---------------------------------------------------------------------------
@njit(fastmath=False)
def _mirror_r(qq, margin, sign_bot, sign_top):
    """r 方向のゴーストセルを鏡像で埋める.

    ``sign = +1`` で対称 (境界で法線微分ゼロ), ``-1`` で反対称
    (境界で値ゼロ). 境界は**セル境界面**にあるので, 鏡像は
    ``q[margin-1-k] = sign * q[margin+k]`` になる.
    """
    ixg, jxg = qq.shape
    for k in range(margin):
        for j in range(jxg):
            qq[margin - 1 - k, j] = sign_bot*qq[margin + k, j]
            qq[ixg - margin + k, j] = sign_top*qq[ixg - margin - 1 - k, j]


@njit(fastmath=False)
def _mirror_th(qq, margin, sign):
    """theta 方向 (両極) のゴーストセルを鏡像で埋める."""
    ixg, jxg = qq.shape
    for k in range(margin):
        for i in range(ixg):
            qq[i, margin - 1 - k] = sign*qq[i, margin + k]
            qq[i, jxg - margin + k] = sign*qq[i, jxg - margin - 1 - k]


@njit(fastmath=False)
def apply_radial_bc(ro1, vrr, vth, om1, se1, rr, ro0, margin,
                    om1_bottom_dirichlet, mass_flux_bc):
    """動径境界の条件を適用する (Rempel 2005 §2.6).

    ==========  =========================  ================================
    変数        条件                       実装
    ==========  =========================  ================================
    ``ro1``     対称                       :math:`\\rho_1` の鏡像
    ``vrr``     閉境界 :math:`v_r=0`       質量フラックスを反対称
    ``vth``     stress-free                :math:`v_\\theta/r` を対称にする
    ``om1``     下: 剛体回転 or stress-free  反対称 or 対称
                上: stress-free            対称
    ``se1``     :math:`\\partial s_1/\\partial r=0`  対称
    ==========  =========================  ================================

    :math:`v_\\theta` の stress-free について
    ------------------------------------------
    stress-free は :math:`R_{r\\theta}=0`, すなわち
    :math:`r\\partial_r(v_\\theta/r) + r^{-1}\\partial_\\theta v_r = 0`.
    閉境界では :math:`v_r=0` なので :math:`\\partial_\\theta v_r=0` となり,
    条件は :math:`\\partial_r(v_\\theta/r)=0` に帰着する. したがって
    :math:`v_\\theta` そのものではなく :math:`v_\\theta/r` を対称にする
    (齋藤 2024 コードの ``symbc_x(vynx)`` と同じ処理).

    ``om1_bottom_dirichlet``
    -------------------------
    True にすると下部境界で :math:`\\Omega_1=0` を課す. これは Rempel が
    タコクラインを強制するために使っている条件で, **系は角運動量について
    閉じなくなる** (境界を通して粘性フラックスが出入りする).
    False なら stress-free になり, 系は閉じて :math:`\\int q_L` が
    machine precision で保存する (齋藤 2024 と同じ).
    """
    _mirror_r(ro1, margin, 1.0, 1.0)
    _mirror_r(se1, margin, 1.0, 1.0)

    # v_r: 反対称にするのは **速度ではなく質量フラックス**
    # rho_0 r^2 v_r. 境界を横切る質量流束をゼロにするのが条件なので、
    # 保存量 q_mr = r^2 sin(theta) rho_0 v_r の鏡像を取るのが正しい。
    #
    # v_r をそのまま反対称にすると、rho_0 が 1 セルで e^(-dr/H_p) 倍
    # 変わる分だけ質量流束が合わなくなる。この不整合は **rho_0 の変化が
    # 最も急な上部境界に集中**し、そこから格子スケールの境界層を生やす。
    # 実測 (2026-08-21): v_r の 2 セル振動が境界に向かって単調に増え、
    # 最外セルで滑らかな成分の 1.8 倍に達していた。解像度を上げても
    # 4 次ハイパー拡散を入れても取れなかったのはこれが理由。
    ixg, jxg = vrr.shape
    if mass_flux_bc:
        for k in range(margin):
            i_in = margin + k
            i_gh = margin - 1 - k
            f = (ro0[i_in]*rr[i_in]*rr[i_in]
                 / (ro0[i_gh]*rr[i_gh]*rr[i_gh]))
            for j in range(jxg):
                vrr[i_gh, j] = -vrr[i_in, j]*f
            i_in = ixg - margin - 1 - k
            i_gh = ixg - margin + k
            f = (ro0[i_in]*rr[i_in]*rr[i_in]
                 / (ro0[i_gh]*rr[i_gh]*rr[i_gh]))
            for j in range(jxg):
                vrr[i_gh, j] = -vrr[i_in, j]*f
    else:
        _mirror_r(vrr, margin, -1.0, -1.0)
    _mirror_r(om1, margin, -1.0 if om1_bottom_dirichlet else 1.0, 1.0)

    # v_theta は v_theta/r を対称にする
    ixg, jxg = vth.shape
    for k in range(margin):
        for j in range(jxg):
            i_in = margin + k
            i_gh = margin - 1 - k
            vth[i_gh, j] = vth[i_in, j]*rr[i_gh]/rr[i_in]
            i_in = ixg - margin - 1 - k
            i_gh = ixg - margin + k
            vth[i_gh, j] = vth[i_in, j]*rr[i_gh]/rr[i_in]


@njit(fastmath=False)
def apply_polar_bc(ro1, vrr, vth, om1, se1, margin):
    """極 (theta = 0, pi) の対称性条件を適用する.

    軸対称なので極は座標特異点ではなく単なる対称面. スカラーと
    :math:`v_r`, :math:`\\Omega_1` は対称, :math:`v_\\theta` は反対称.
    S2MFD は極をセル境界面に置くので, 鏡像で正しく埋まる.
    """
    _mirror_th(ro1, margin, 1.0)
    _mirror_th(vrr, margin, 1.0)
    _mirror_th(vth, margin, -1.0)
    _mirror_th(om1, margin, 1.0)
    _mirror_th(se1, margin, 1.0)


# ---------------------------------------------------------------------------
# ソルバ
# ---------------------------------------------------------------------------
class DynamicSolver:
    """保存形の平均場流体ソルバ.

    Parameters
    ----------
    cfg : S2MFD.Cfg
    grid : S2MFD.Grid
    strat : S2MFD.stratification.Stratification
    setup : S2MFD.Setup
        ``build_turbulent_transport`` と ``build_lambda`` が実行済みであること.
    """

    def __init__(self, cfg, grid, strat, setup):
        self.cfg = cfg
        self.grid = grid
        self.strat = strat
        self.setup = setup
        self.m = grid.margin
        shape = (grid.ixg, grid.jxg)
        self.shape = shape

        self.magnetic = (getattr(cfg, 'dynamics', 'kinematic') == 'dynamic')
        #: 動径運動量の磁気圧勾配 (磁気浮力) を含めるか。Rempel (2006) 表1
        #: の列 4/6/8 はこれを False にした「magnetic buoyancy off」の解。
        self.magnetic_buoyancy = bool(getattr(cfg, 'magnetic_buoyancy', True))
        self.om1_bottom_dirichlet = (
            getattr(cfg, 'angmom_bottom_bc', 'uniform_rotation')
            == 'uniform_rotation')
        self.use_artdif = getattr(cfg, 'artificial_diffusion', True)
        if not self._top_is_pole and self.m < 2 and self.use_artdif:
            warnings.warn(
                f"北半球のみ (thmax = pi/2) を margin={self.m} で解いています。"
                "SLD のリミタ (sld_flux_th) は境界面の 2 セル先を参照するので、"
                "赤道でゴーストが 1 層だと片側差分に落ち、全球計算と違う"
                "拡散フラックスになります。margin=2 以上にしてください。",
                UserWarning, stacklevel=2)
        # Rempel (2014) の SLD パラメタ (R2D2 と同じ既定値)
        self.sld_fh = getattr(cfg, 'sld_fh', 2.0)   # 論文の h
        self.sld_ep = getattr(cfg, 'sld_ep', 2.0)   # 一般化 minmod の epsilon
        # 特性速度 |v| + v_A + cs_factor*c_s,eff の音速係数 (R2D2 は 0.3)
        self.cs_factor = getattr(cfg, 'sld_cs_factor', 0.3)
        # Rempel (2014) の 4 次ハイパー拡散の係数。移流速度に比例し、
        # 動径方向のみ、背景勾配に隠れた格子スケール振動を潰す。
        # 論文に数値の指定はないので実測で決める。
        self.hyper_h4 = getattr(cfg, 'hyper_h4', 0.0)
        # 緯度平均プロファイルへの拡散係数 (nu_t 単位)。
        # 緯度平均でしか見えない格子スケール成分を潰す。標的が限定されて
        # いるので弱くてよく、物理的な緯度平均 v_r はほぼゼロなので解を削らない。
        self.mean_diff_frac = getattr(cfg, 'mean_profile_diffusion', 0.0)
        self.consistent_advection = getattr(cfg, 'consistent_advection', False)
        #: 上下境界で v_r ではなく **質量フラックス rho_0 r^2 v_r** を
        #: 反対称にするか。既定 True。False にすると旧来の v_r 反対称。
        self.mass_flux_bc = bool(getattr(cfg, 'mass_flux_bc', True))

        # プリミティブ変数
        self.ro1 = np.zeros(shape)
        self.vrr = np.zeros(shape)
        self.vth = np.zeros(shape)
        self.om1 = np.zeros(shape)
        self.se1 = np.zeros(shape)
        self.pr1 = np.zeros(shape)

        # 磁場 (dynamic モードで外から与えられる)
        self.brr = np.zeros(shape)
        self.bth = np.zeros(shape)
        self.bph = np.zeros(shape)

        self.work = hydro.HydroWork(grid)
        self.izeta2 = 1.0/strat.zeta**2

        # 人工拡散のヤコビアン係数 (面上)
        self._build_artdif_coefficients()

        # rhs の作業配列 (毎回確保せず使い回す)
        self._rhs_bufs = tuple(np.zeros(shape) for _ in range(9))
        self._zero = np.zeros(shape)
        # 時間積分の作業配列 (保存量 5 つ x 3 段)
        self._q0 = tuple(np.zeros(shape) for _ in range(5))
        self._q1 = tuple(np.zeros(shape) for _ in range(5))
        self._qn = tuple(np.zeros(shape) for _ in range(5))

        # 診断用の累積量
        # 下部境界を通って流入した角運動量の時間積分 [erg s]。
        # angmom_bottom_bc='uniform_rotation' では系が閉じないので、
        # 「保存する」代わりに「収支が厳密に合う」ことを検証するために使う。
        self.boundary_angmom_flux = 0.0
        self._bflux = np.zeros(grid.jxg)
        # 人工拡散による角運動量の散逸率 [erg/s]。Rempel のエネルギー収支
        # (式 20) には現れない項だが、本実装では無視できない大きさになりうる
        # ので、収支が閉じるかどうかを見るために別途記録する。
        self.artificial_dissipation = 0.0

    # -- 初期化 ----------------------------------------------------------
    def _build_artdif_coefficients(self):
        """人工拡散フラックスに掛けるヤコビアンを面上で用意する."""
        s = self.strat

        def face_r(arr):
            out = np.zeros(self.shape)
            out[1:] = 0.5*(arr[1:] + arr[:-1])
            return out

        def face_th(arr):
            out = np.zeros(self.shape)
            out[:, 1:] = 0.5*(arr[:, 1:] + arr[:, :-1])
            return out

        self.jacL_r, self.jacL_th = face_r(s.JL), face_th(s.JLY)
        self.jacV_r, self.jacV_th = face_r(s.JV), face_th(s.JVY)
        # 質量の theta フラックスは r の冪が 1 つ低い (r sin(theta))
        self.jacM_r, self.jacM_th = face_r(s.JM), face_th(s.RSIN)
        self.iJM = s.iJM
        # 磁場用 (スカラー拡散の幾何係数):
        #   d/dt u = (1/(r^2 sin th)) [ d_r(r^2 sin th d_r u) + d_th(sin th d_th u) ]
        self.jacB_r = face_r(s.JM)
        self.jacB_th = face_th(self.grid.sinTH)
        self.iJB = s.iJM
        # 特性速度は毎ステップ更新する
        self.csp_r = np.zeros(self.shape)
        self.csp_th = np.zeros(self.shape)
        # ハイパー拡散用の移流速度 (面上)
        self.vadv_r = np.zeros(self.shape)
        # 磁場フィルタ用の特性速度 (音速を含まない)
        self.cspB_r = np.zeros(self.shape)
        self.cspB_th = np.zeros(self.shape)
        # 緯度平均拡散の 1 次元作業配列
        self._w1 = np.zeros(self.shape[0])
        self._p1 = np.zeros(self.shape[0])
        self._d1 = np.zeros(self.shape[0])


    def _update_characteristic_speed(self):
        """人工拡散に使う特性速度を面上で作る.

        .. math::
           c = |\\boldsymbol{v}| + h_{c_s}\\,c_{s,\\rm eff} + v_A

        **流れ・抑制した音速・アルヴェン速度の 3 つの和**を取り, 音速には
        :math:`h_{c_s}=0.1\\sim0.3` 程度の係数を掛ける
        (``cfg.sld_cs_factor``, 既定 0.2).

        音速に係数を掛ける理由
        ----------------------
        抑制後とはいえ音速は流れより 2 桁以上速い
        (:math:`c_{s,\\rm eff}\\sim2.6\\times10^5` に対し
        :math:`|v|\\sim10^3` cm/s). そのまま使うと人工拡散の実効拡散係数

        .. math:: \\kappa_{\\rm SLD} \\sim \\tfrac12 h\\,c\\,\\Delta x

        が大きくなりすぎ, **モデルが依存している「拡散を意図的に小さくした
        領域」を潰してしまう**:

        * オーバーシュート層の乱流熱伝導 :math:`\\kappa_t` は対流層値の
          0.2% (:math:`\\sim5\\times10^{10}`). ここにエントロピー摂動を
          溜めて Taylor-Proudman 制約を破るのが Rempel (2005) の中心的な機構
        * 放射層の粘性 :math:`\\nu_{\\rm dif}` は対流層値の 2%
          (:math:`6\\times10^{10}`). 下部境界の :math:`\\Omega_1=0` との間に
          タコクライン (粘性せん断層) を作るのが要

        係数なしだと :math:`\\kappa_{\\rm SLD}\\sim4.7\\times10^{11}` で
        どちらの 7-10 倍にもなり, 差動回転が論文の 1/10 に留まっていた
        (``doc/dev_records/2026-08-20_rempel2006_worklog.md`` §5b).
        """
        s = self.strat
        cc = (np.sqrt(self.vrr**2 + self.vth**2)
              + self.cs_factor*s.cs_eff[:, None])
        if self.magnetic:
            rho = np.maximum(s.ro0[:, None] + self.ro1, 1e-30)
            cc = cc + np.sqrt((self.brr**2 + self.bth**2 + self.bph**2)
                              / (4.0*np.pi*rho))
        self.csp_r[1:] = 0.5*(cc[1:] + cc[:-1])
        self.csp_th[:, 1:] = 0.5*(cc[:, 1:] + cc[:, :-1])
        vv = np.sqrt(self.vrr**2 + self.vth**2)
        self.vadv_r[1:] = 0.5*(vv[1:] + vv[:-1])
        # 磁場フィルタ用の特性速度は**流れだけ**。
        # 誘導方程式で磁場を運ぶのは流れであって音波ではない。
        # アルヴェン波も入れない: 運動学的ダイナモでは運動方程式を解かないので
        # そもそもアルヴェン波が存在しないし、力学モードでも磁場の輸送速度は
        # 流れである (アルヴェン波は運動量方程式との結合で現れる)。
        # 音速を入れると、運動学的ランでフィルタの実効拡散係数 (1/2)c*dx が
        # 磁気拡散 eta を桁で上回り CFL を破る (実測 1.3e13 対 eta=1e12)。
        self.cspB_r[1:] = 0.5*(vv[1:] + vv[:-1])
        self.cspB_th[:, 1:] = 0.5*(vv[:, 1:] + vv[:, :-1])

    # -- 状態の変換 -------------------------------------------------------
    def conserved(self, out=None):
        """現在のプリミティブ変数から保存量を作る.

        ``out`` に配列の組を渡すとその場に書き込む (時間積分で使い回す).
        """
        s = self.strat
        if out is None:
            return (s.JM*self.ro1, s.JV*self.vrr, s.JV*self.vth,
                    s.JL*self.om1, self.se1.copy())
        _mul_into(out[0], s.JM, self.ro1)
        _mul_into(out[1], s.JV, self.vrr)
        _mul_into(out[2], s.JV, self.vth)
        _mul_into(out[3], s.JL, self.om1)
        out[4][:] = self.se1
        return out

    def set_primitive_from_conserved(self, qq):
        """保存量からプリミティブ変数を復元し, 境界条件を適用する."""
        q_ro, q_mr, q_mt, q_om, se1 = qq
        s = self.strat
        m = self.m
        sl = (slice(m, self.grid.ixg - m), slice(m, self.grid.jxg - m))
        self.ro1[sl] = (q_ro*s.iJM)[sl]
        self.vrr[sl] = (q_mr*s.iJV)[sl]
        self.vth[sl] = (q_mt*s.iJV)[sl]
        self.om1[sl] = (q_om*s.iJL)[sl]
        self.se1[:] = se1
        apply_radial_bc(self.ro1, self.vrr, self.vth, self.om1, self.se1,
                        self.grid.rr, s.ro0, m, self.om1_bottom_dirichlet,
                        self.mass_flux_bc)
        apply_polar_bc(self.ro1, self.vrr, self.vth, self.om1, self.se1, m)
        self.pr1[:] = s.pressure_perturbation(self.ro1, self.se1)

    # -- 右辺 -------------------------------------------------------------
    def rhs(self):
        """現在のプリミティブ変数から時間微分を返す.

        Returns
        -------
        tuple
            ``(dq_ro, dq_mr, dq_mt, dq_om, dse1)``
        """
        cfg, grid, s, st = self.cfg, self.grid, self.strat, self.setup
        m = self.m
        w = self.work
        z = self._zero

        # 作業配列は使い回す。毎 substep で 9 本を np.zeros すると
        # 配列確保とゼロ埋めだけで rhs の 2 割以上を食う。
        (dq_ro, dq_mr, dq_mt, dq_om, dse1,
         ds_mr, ds_mt, ds_om, heat) = self._rhs_bufs
        for a in self._rhs_bufs:
            a[:] = 0.0

        # --- 質量 --------------------------------------------------------
        hydro.mass_rhs(dq_ro, self.vrr, self.vth, s.JV, s.JVY, self.izeta2,
                       grid.drr, grid.wfm, grid.dth, m, w.ffr, w.ffth, w.cen, self._top_is_pole)

        # --- 子午面運動量 (移流 + 幾何源項 + 圧力 + 浮力 + ローレンツ) ----
        hydro.momentum_rhs(dq_mr, dq_mt, self.vrr, self.vth, self.om1,
                           self.ro1, self.pr1, self.brr, self.bth, self.bph,
                           s.JV, s.JVY, s.JM, s.RSIN, grid.RR, grid.sinTH,
                           grid.cosTH, s.ro0, s.gr, cfg.om0,
                           grid.drr, grid.drr2, grid.wfm, grid.dth, m,
                           self.magnetic,
                           w.ffr, w.ffth, w.cen, self.magnetic_buoyancy, self._top_is_pole)

        # --- 角運動量 (移流 + Maxwell、および分離したレイノルズ応力) ------
        hydro.angular_momentum_rhs(
            dq_om, self.om1, self.vrr, self.vth,
            self.brr, self.bth, self.bph,
            s.JL, s.JLY, s.JV, s.JVY, s.W2, grid.RR, grid.RRm,
            grid.sinTH, grid.sinTHm, s.ro0, s.ro0m,
            st.lam_rp, st.lam_tp, cfg.om0,
            st.nu_dif, st.nu_dif_m, st.nu_lam, st.nu_lam_m,
            grid.drr, grid.drrm, grid.wfm, grid.dth, m, self.magnetic,
            self.consistent_advection, self.om1_bottom_dirichlet,
            ds_om, heat, self._bflux, w.ffr, w.ffth, w.cen, self._top_is_pole)

        # --- 子午面の粘性 (散逸項として分離) -----------------------------
        hydro.viscous_meridional_rhs(ds_mr, ds_mt, self.vrr, self.vth,
                                     grid.rr, grid.sinTH, grid.cosTH, s.ro0,
                                     st.nu_dif, st.nu_dif_m,
                                     grid.drr, grid.drrm, grid.drr2, grid.wfm,
                                     grid.dth, m, w.ffr, w.ffth, self._top_is_pole)

        # --- 人工拡散 (これも散逸項) -------------------------------------
        if self.use_artdif:
            self._update_characteristic_speed()
            heat_before = heat.copy()
            artdif.sld_diffuse_work(ds_om, heat, self.om1,
                                    self.jacL_r, self.jacL_th,
                                    self.csp_r, self.csp_th,
                               self.sld_fh, self.sld_ep,
                                    grid.drr, grid.drrm, grid.dth, m,
                                    w.ffr, w.ffth, self._top_is_pole)
            # 子午面速度はベクトル成分なので、theta 掃引で基底が回る分の
            # 幾何項が要る (v_r と v_theta が混ざる)
            artdif.sld_diffuse_meridional(
                ds_mr, ds_mt, self.vrr, self.vth,
                self.jacV_r, self.jacV_th, self.csp_r, self.csp_th,
                self.sld_fh, self.sld_ep, grid.drr, grid.drrm, grid.dth, m,
                w.ffr, w.ffth, w.ffr2, w.ffth2, self._top_is_pole)
            # 密度にも掛ける (音波の格子スケール振動を抑える)。保存量は
            # ∫ζ²ρ1 dV なので、連続の式と同じく発散に 1/ζ² を掛ける
            artdif.sld_diffuse_scaled(dq_ro, self.ro1, self.jacM_r,
                                      self.jacM_th, self.csp_r, self.csp_th,
                                      self.sld_fh, self.sld_ep,
                                      grid.drr, grid.drrm, grid.dth, m,
                                      self.izeta2, w.ffr, w.ffth, self._top_is_pole)

            # 人工拡散が角運動量から抜いたエネルギーを記録する (診断用)。
            # heat は -F.grad(Omega1) なので、その体積積分が散逸率になる。
            # heat は -F.grad(Omega1) = 流れから抜けたエネルギー (加熱側が正)。
            # その体積積分がそのまま散逸率。q_L はヤコビアンを吸収済みなので
            # sum * drr * dth に方位角の 2pi を掛ければ体積積分になる。
            # 方位角因子は EnergyBudget と同じ規約 (半球なら全球換算で 4pi)
            azim = 2.0*np.pi if self._top_is_pole else 4.0*np.pi
            self.artificial_dissipation = azim*float(
                ((heat - heat_before)[m:grid.ixg - m, m:grid.jxg - m]
                 * grid.drr[m:grid.ixg - m, None]).sum())*grid.dth

            # エントロピーにも掛ける。ここを忘れると s1 の格子ノイズが
            # 減衰せず、p1 = p0(gamma*rho1/rho0 + s1) を通して rho0 の小さい
            # 対流層上部で巨大な加速を生み、計算が壊れる
            artdif.sld_diffuse_primitive(dse1, self.se1, self.jacM_r,
                                         self.jacM_th, self.iJM,
                                         self.csp_r, self.csp_th,
                                         self.sld_fh, self.sld_ep,
                                         grid.drr, grid.drrm, grid.dth, m,
                                         w.ffr, w.ffth, self._top_is_pole)

            # --- 4 次ハイパー拡散 (Rempel 2014) ---------------------------
            # 背景勾配があると SLD のリミタが「単調」と判断してしまい、
            # その上に乗った格子スケールの振動を素通りさせる。4 階微分なら
            # 線形・2 次の背景を見ないので、それだけを選択的に潰せる。
            # 動径方向のみ、移流速度に比例。対象は Rempel と同じく
            # 密度・重力方向の速度・内部エネルギー (ここでは rho1, v_r, s1)。
            # 緯度平均プロファイルへの拡散 (緯度平均でしか見えない成分用)
            if self.mean_diff_frac > 0.0:
                kap = self.mean_diff_frac*float(np.max(st.nu_dif))
                artdif.mean_profile_diffuse_r(
                    ds_mr, self.vrr, s.JV, kap, grid.drr, grid.drrm, m,
                    self._w1, self._p1, self._d1)
                artdif.mean_profile_diffuse_r_primitive(
                    dse1, self.se1, s.JM, s.iJM, kap, grid.drr, grid.drrm, m,
                    self._w1, self._p1, self._d1)

            if self.hyper_h4 > 0.0:
                artdif.hyper_diffuse_r(ds_mr, self.vrr, self.jacV_r,
                                       self.vadv_r, self.hyper_h4,
                                       grid.drr, grid.dth, m, w.ffr, w.ffth)
                artdif.hyper_diffuse_r_primitive(
                    dse1, self.se1, self.jacM_r, self.iJM, self.vadv_r,
                    self.hyper_h4, grid.drr, m, w.ffr)
                # 密度は保存量なので RSST の 1/zeta^2 を掛ける
                artdif.hyper_flux_r(self.ro1, self.jacM_r, self.vadv_r,
                                    self.hyper_h4, m, w.ffr)
                cons.zero_boundary_faces_r(w.ffr, m)
                for jj in range(grid.jxg):
                    for ii in range(grid.ixg):
                        w.ffth[ii, jj] = 0.0
                cons.add_flux_divergence_scaled(dq_ro, w.ffr, w.ffth,
                                                grid.drr, grid.dth, m,
                                                self.izeta2)

        # --- エントロピー -------------------------------------------------
        hydro.entropy_rhs(dse1, self.se1, self.vrr, self.vth, s.ro0, s.tm0,
                          s.pr0, s.hp, s.delta, st.kappa_t, st.kappa_t_m,
                          s.JM, s.iJM, grid.rr, grid.sinTH, grid.sinTHm,
                          s.gamma, grid.drr, grid.drrm, grid.drr2, grid.wfm,
                          grid.dth, m, w.ffr, w.ffth, self._top_is_pole)
        # 散逸したエネルギーをエントロピーに戻す (係数はちょうど 1)
        hydro.add_dissipative_heating(dse1, heat, ds_mr, ds_mt,
                                      self.vrr, self.vth,
                                      s.pr0, s.iJM, s.gamma, m)
        if self.magnetic:
            hydro.add_ohmic_heating(dse1, self.brr, self.bth, self.bph,
                                    st.et, s.pr0, grid.RR, grid.sinTH,
                                    s.gamma, grid.drr2, grid.dth, m)

        # --- 散逸項を本体に合流 ------------------------------------------
        dq_mr += ds_mr
        dq_mt += ds_mt
        dq_om += ds_om
        return (dq_ro, dq_mr, dq_mt, dq_om, dse1)

    # -- 時間積分 ---------------------------------------------------------
    def step(self, dt):
        """SSP-RK2 (Heun) で 1 ステップ進める.

        下部境界を通った角運動量も RK2 と同じ重みで積算するので,
        ``sum(q_L) - sum(q_L)_0 == boundary_angmom_flux`` が
        machine precision で成り立つ (境界が開いている設定でも
        「漏れているが勘定は合っている」ことを検証できる).
        """
        q0 = self.conserved(self._q0)
        k1 = self.rhs()
        # 物理セルの範囲だけ足す (ゴーストセルは発散に寄与しない)
        j0, j1 = self.m, self.grid.jxg - self.m
        f1 = self._bflux[j0:j1].sum()*self.grid.dth
        q1 = self._q1
        for a, b, o in zip(q0, k1, q1):
            _rk_predictor(o, a, b, dt)
        self.set_primitive_from_conserved(q1)
        k2 = self.rhs()
        f2 = self._bflux[j0:j1].sum()*self.grid.dth
        qn = self._qn
        for a, b, c, o in zip(q0, q1, k2, qn):
            _rk_corrector(o, a, b, c, dt)
        self.set_primitive_from_conserved(qn)
        # 面フラックスは「流出」向きが正なので、流入分は符号を反転して積算
        self.boundary_angmom_flux += dt*0.5*(f1 + f2)

    # -- 磁場との結合 -----------------------------------------------------
    def sync_to_induction(self):
        """流体の解を誘導方程式カーネルが読む ``setup`` の配列に反映する.

        既存の運動学的ダイナモのカーネル (:func:`S2MFD.physics.time_marching`)
        は流れ場を ``setup.urr``, ``setup.uth``, ``setup.omrr``,
        ``setup.omth`` から読む. 力学モードではこれらが毎ステップ変わるので,
        ソルバの解で上書きする.

        こうすることで**誘導方程式そのものは運動学的ダイナモと同一の
        コードを使う**. Rempel 2006 式 (6)(7) と S2MFD の誘導方程式は
        既に項ごとに一致していることを確認済みなので, 書き直す必要はない
        (``doc/dev_records/2026-08-19_rempel2006_feasibility.md`` §1).

        Ω効果に入るのは :math:`\\partial\\Omega_1/\\partial r` と
        :math:`\\partial\\Omega_1/\\partial\\theta` だが, :math:`\\Omega_0` は
        定数なので :math:`\\Omega=\\Omega_0+\\Omega_1` の微分と同じである.
        """
        from S2MFD.tools import drr2, dth2
        st, grid = self.setup, self.grid
        st.urr = self.vrr
        st.uth = self.vth
        om = self.cfg.om0 + self.om1
        st.om = om
        st.omrr = drr2(om, grid.drr2)
        st.omth = dth2(om, grid.dth)/grid.RR

    def magnetic_filter(self, bph, aph, dt):
        """磁場に人工拡散 (SLD) を掛ける — 誘導方程式の後のフィルタ段.

        Rempel (2014) は SLD を
        :math:`\\{\\log\\rho, v_x, v_y, v_z, \\varepsilon, B_x, B_y, B_z\\}`
        の**全変数**に掛けており, 磁場も例外ではない. 実装も
        「4 次時間積分の 1 ステップを終えた後の独立したフィルタ段」として
        適用すると明記されている:

            "The numerical diffusion scheme is implemented in a dimensional
             split way ... and is applied to the solution in a separate
             filtering step after a full time-step update of our fourth-order
             time integration scheme."

        本実装もそれに倣い, **検証済みの誘導方程式カーネルには一切触れずに**
        その後段で掛ける. これにより運動学的ダイナモ (``dynamics='kinematic'``)
        の経路は完全に無変更のまま保たれる.

        対象は :math:`B_\\varphi` とベクトルポテンシャル :math:`A_\\varphi`.
        :math:`A_\\varphi` を拡散させればポロイダル場が拡散し, しかも
        :math:`\\nabla\\cdot B=0` は自動的に保たれる.

        特性速度は :math:`|v|+v_A` で, **音速を含めない**. 磁場は音波では
        運ばれないためで, 実用上も重要である: 運動学的ダイナモ (流れを固定して
        誘導方程式だけを解く) では :math:`\\Delta t` が音速で決まらないので,
        音速を入れるとフィルタの実効拡散係数 :math:`\\tfrac12 c\\Delta x` が
        磁気拡散 :math:`\\eta_t` を桁で上回って CFL を破る
        (実測: :math:`1.3\\times10^{13}` 対 :math:`\\eta_t=10^{12}`).

        Parameters
        ----------
        dt : float
            フィルタ段で進める時間 (本体のタイムステップと同じ).

        Returns
        -------
        tuple
            フィルタ後の ``(bph, aph)``. 引数の配列がその場で更新される.
        """
        # 人工拡散全体のスイッチとは別に、磁場フィルタだけ切れるようにする。
        # Rempel (2006) には磁場への人工拡散がないので、それが磁場の振幅に
        # どれだけ効いているかを測るための対照実験用。
        if not self.use_artdif or not getattr(self.cfg, 'magnetic_filter', True):
            return bph, aph
        grid, m, w = self.grid, self.m, self.work
        self._update_characteristic_speed()
        for fld in (bph, aph):
            d = self._rhs_bufs[0]
            d[:] = 0.0
            artdif.sld_diffuse_primitive(
                d, np.ascontiguousarray(fld), self.jacB_r, self.jacB_th,
                self.iJB, self.cspB_r, self.cspB_th, self.sld_fh, self.sld_ep,
                grid.drr, grid.drrm, grid.dth, m, w.ffr, w.ffth, self._top_is_pole)
            fld += dt*d
        return bph, aph

    def poloidal_from_potential(self, aph):
        """ベクトルポテンシャルからポロイダル磁場 (B_r, B_theta) を作る.

        :func:`~S2MFD.physics.physics_core.poloidal_mag` の第 4 引数は
        **``grid.drr2`` (2 セル幅)** である。``grid.drr`` (1 セル幅) を
        渡すと :math:`B_\theta` がちょうど 2 倍になる。

        実際 ``run_paris/dynamo7.py`` と ``ana/ana_common.py`` が
        ``grid.drr`` を渡しており、**ローレンツ力に使う** :math:`B_\theta`
        が 2 倍になっていた (2026-08-23 に発見)。誘導方程式はカーネル内部で
        自前に :math:`B_p` を作るので影響を受けず、運動学的ランは正しかった
        が、非運動学的ランはマクスウェル応力が過大になり低い磁場で飽和して
        いた (論文比 0.6)。

        引数の取り違えを二度と起こさないよう、呼び出しはこのメソッドに
        集約すること。
        """
        return hydro_poloidal_mag(aph, self.grid.RR, self.grid.sinTH,
                                  self.grid.drr2, self.grid.dth)

    def set_magnetic_field(self, brr, bth, bph):
        """ローレンツ力に使う磁場を外から与える."""
        self.brr[:] = brr
        self.bth[:] = bth
        self.bph[:] = bph

    @property
    def _top_is_pole(self):
        """theta 方向の上端が極か (True) 赤道か (False).

        北半球のみを解く場合 (thmax = pi/2) は上端が赤道なので、拡散
        フラックスをそこでゼロにしてはいけない
        (:func:`S2MFD.physics.conservative.zero_boundary_faces_th` 参照)。
        """
        thmax = getattr(self.cfg, 'thmax', np.pi)
        return abs(thmax - 0.5*np.pi) > 1.0e-9

    def linear_stability(self, dt=None):
        """現在の設定での max ln|G| と最悪半径を返す.

        0 以下なら線形安定. 正なら非線形リミタの助けに頼っている
        (:mod:`S2MFD.physics.stability` の docstring 参照).
        """
        if dt is None:
            dt = self.cfl_dt()
        return stability.max_log_growth(self.cfg, self.grid, self.strat,
                                        self.setup, dt)

    def neutral_cfl_safety(self):
        """線形安定に保てる ``cfl_safety`` の上限を返す."""
        cfg = self.cfg
        saved = getattr(cfg, 'cfl_safety', 0.2)

        def dt_of(S):
            cfg.cfl_safety = S
            return self.cfl_dt()
        try:
            return stability.neutral_cfl_safety(cfg, self.grid, self.strat,
                                                self.setup, dt_of)
        finally:
            cfg.cfl_safety = saved

    def cfl_dt(self):
        """CFL 条件から許容タイムステップを返す.

        拡散の安定条件は「方程式ごとに、その方程式に効く拡散係数を**足して**
        から、方程式間で max を取る」
        -------------------------------------------------------------------
        運動量方程式には :func:`~S2MFD.physics.hydro.viscous_meridional_rhs`
        の :math:`\\nu_{\\rm dif}` と
        :func:`~S2MFD.physics.artdif.sld_diffuse_meridional` の人工拡散が
        **同じステップで同時に**効くので, 実効拡散係数はその**和**になる.
        エントロピーは :math:`\\kappa_t` + 人工拡散, 誘導方程式は
        :math:`\\eta_t` + 磁場フィルタの人工拡散.

        全部まとめて ``max`` を取ると過小評価になり, 物理拡散と人工拡散が
        同程度の大きさになる領域で安定余裕が落ちる. 実測 (論文設定 108x72)
        では ``sld_cs_factor = 0.20`` だけが発散し, より拡散の弱い 0.15 と
        強い 0.30 は安定という非単調な結果になっていた. dt を決める制約が
        cs <= 0.2 では音波, cs >= 0.3 では SLD 拡散に切り替わるため, 安全
        余裕が 0.20 で最小になっていたのが原因
        (``doc/dev_records/2026-08-22_cfl_safety_vs_artdif.md``).

        磁場フィルタの特性速度は**流れだけ**なので (音速もアルヴェン速度も
        入れない — :meth:`magnetic_filter` 参照), 誘導方程式側の人工拡散は
        流体側より小さくなる. 別々に評価する.
        """
        cfg, s, st = self.cfg, self.strat, self.setup
        k_sld = k_sld_b = 0.0
        if self.use_artdif:
            # 人工拡散も陽解法なので安定条件に入れる
            self._update_characteristic_speed()
            k_sld = artdif.sld_diffusivity_max(
                self.csp_r, self.csp_th,
                self.grid.drr, self.grid.dth, self.grid.rr, self.m)
            if self.magnetic:
                k_sld_b = artdif.sld_diffusivity_max(
                    self.cspB_r, self.cspB_th,
                    self.grid.drr, self.grid.dth, self.grid.rr, self.m)
        # 方程式ごとに和を取ってから max
        kappa_max = max(float(np.max(st.nu_dif)) + k_sld,      # 運動量
                        float(np.max(st.kappa_t)) + k_sld)     # エントロピー
        if self.magnetic:
            kappa_max = max(kappa_max,
                            float(np.max(st.et)) + k_sld_b)    # 誘導
        dt = hydro.cfl_dt(self.vrr, self.vth, self.brr, self.bth, self.bph,
                          s.ro0, self.ro1, s.cs_eff, self.grid.rr,
                          self.grid.drr, self.grid.dth, kappa_max, self.m,
                          getattr(cfg, 'cfl_safety', 0.2), self.magnetic)
        self._warn_if_linearly_unstable(dt)
        return dt

    def _warn_if_linearly_unstable(self, dt):
        """線形不安定な設定なら 1 度だけ警告する.

        中央差分 + SSP-RK2 は拡散がないと**無条件不安定**なので、人工拡散を
        弱めたまま安全率を上げると非線形リミタの助けに頼ることになる
        (:mod:`S2MFD.physics.stability` の docstring 参照)。
        """
        if getattr(self, '_stability_checked', False):
            return
        self._stability_checked = True
        if not self.use_artdif:
            return
        cfg = self.cfg
        try:
            g, r = self.linear_stability(dt)
        except Exception:                     # 診断なので失敗しても止めない
            return
        if g <= 0.0:
            return
        try:
            S0 = self.neutral_cfl_safety()
        except Exception:
            S0 = float('nan')
        warnings.warn(
            f"線形不安定な設定です (max ln|G| = {g:+.2e}, 最悪半径 {r:.4f} R)。"
            f"sld_cs_factor={getattr(cfg, 'sld_cs_factor', 0.3)} なら "
            f"cfl_safety <= {S0:.3f} にしてください "
            f"(現在 {getattr(cfg, 'cfl_safety', 0.2)})。"
            "中央差分 + SSP-RK2 は拡散がないと無条件不安定なので、"
            "人工拡散を弱めるときは安全率も下げる必要があります。",
            UserWarning, stacklevel=3)
