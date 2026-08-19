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
import numpy as np
from numba import njit

from S2MFD.physics import hydro, artdif

__all__ = ['DynamicSolver', 'apply_radial_bc', 'apply_polar_bc']


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
def apply_radial_bc(ro1, vrr, vth, om1, se1, rr, margin, om1_bottom_dirichlet):
    """動径境界の条件を適用する (Rempel 2005 §2.6).

    ==========  =========================  ================================
    変数        条件                       実装
    ==========  =========================  ================================
    ``ro1``     対称                       :math:`\\rho_1` の鏡像
    ``vrr``     閉境界 :math:`v_r=0`       反対称
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
    _mirror_r(vrr, margin, -1.0, -1.0)
    _mirror_r(se1, margin, 1.0, 1.0)
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
        self.om1_bottom_dirichlet = (
            getattr(cfg, 'angmom_bottom_bc', 'uniform_rotation')
            == 'uniform_rotation')
        self.use_artdif = getattr(cfg, 'artificial_diffusion', True)
        self.sld_coef = getattr(cfg, 'sld_coefficient', 1.0)
        self.sld_floor = getattr(cfg, 'sld_floor', 0.1)
        self.consistent_advection = getattr(cfg, 'consistent_advection', False)

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

        # 診断用の累積量
        self.boundary_angmom_flux = 0.0   # 下部境界を通った角運動量 (時間積分)

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
        # 特性速度は毎ステップ更新する
        self.csp_r = np.zeros(self.shape)
        self.csp_th = np.zeros(self.shape)

    def _update_characteristic_speed(self):
        """人工拡散に使う特性速度 |v| + c_s,eff を面上で作る."""
        s = self.strat
        cc = np.sqrt(self.vrr**2 + self.vth**2) + s.cs_eff[:, None]
        if self.magnetic:
            rho = np.maximum(s.ro0[:, None] + self.ro1, 1e-30)
            cc = cc + np.sqrt((self.brr**2 + self.bth**2 + self.bph**2)
                              / (4.0*np.pi*rho))
        self.csp_r[1:] = 0.5*(cc[1:] + cc[:-1])
        self.csp_th[:, 1:] = 0.5*(cc[:, 1:] + cc[:, :-1])

    # -- 状態の変換 -------------------------------------------------------
    def conserved(self):
        """現在のプリミティブ変数から保存量を作る."""
        s = self.strat
        return (s.JM*self.ro1, s.JV*self.vrr, s.JV*self.vth, s.JL*self.om1,
                self.se1.copy())

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
                        self.grid.rr, m, self.om1_bottom_dirichlet)
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
        z = np.zeros(self.shape)

        dq_ro = np.zeros(self.shape)
        dq_mr = np.zeros(self.shape)
        dq_mt = np.zeros(self.shape)
        dq_om = np.zeros(self.shape)
        dse1 = np.zeros(self.shape)
        # 散逸項だけを集める配列 (エントロピーへ渡す)
        ds_mr = np.zeros(self.shape)
        ds_mt = np.zeros(self.shape)
        ds_om = np.zeros(self.shape)

        # --- 質量 --------------------------------------------------------
        hydro.mass_rhs(dq_ro, self.vrr, self.vth, s.JV, s.JVY, self.izeta2,
                       grid.drr, grid.dth, m, w.ffr, w.ffth, w.cen)

        # --- 子午面運動量 (移流 + 幾何源項 + 圧力 + 浮力 + ローレンツ) ----
        hydro.momentum_rhs(dq_mr, dq_mt, self.vrr, self.vth, self.om1,
                           self.ro1, self.pr1, self.brr, self.bth, self.bph,
                           s.JV, s.JVY, s.JM, s.RSIN, grid.RR, grid.sinTH,
                           grid.cosTH, s.ro0, s.gr, cfg.om0,
                           grid.drr, grid.dth, m, self.magnetic,
                           w.ffr, w.ffth, w.cen)

        # --- 角運動量 (移流 + Maxwell、および分離したレイノルズ応力) ------
        hydro.angular_momentum_rhs(
            dq_om, self.om1, self.vrr, self.vth,
            self.brr, self.bth, self.bph,
            s.JL, s.JLY, s.JV, s.JVY, s.W2, grid.RR, grid.RRm,
            grid.sinTH, grid.sinTHm, s.ro0, s.ro0m,
            st.lam_rp, st.lam_tp, cfg.om0,
            st.nu_dif, st.nu_dif_m, st.nu_lam, st.nu_lam_m,
            grid.drr, grid.dth, m, self.magnetic,
            self.consistent_advection, ds_om, w.ffr, w.ffth, w.cen)

        # --- 子午面の粘性 (散逸項として分離) -----------------------------
        hydro.viscous_meridional_rhs(ds_mr, ds_mt, self.vrr, self.vth,
                                     grid.rr, grid.sinTH, grid.cosTH, s.ro0,
                                     st.nu_dif, st.nu_dif_m,
                                     grid.drr, grid.dth, m, w.ffr, w.ffth)

        # --- 人工拡散 (これも散逸項) -------------------------------------
        if self.use_artdif:
            self._update_characteristic_speed()
            artdif.sld_diffuse(ds_om, self.om1, self.jacL_r, self.jacL_th,
                               self.csp_r, self.csp_th, self.sld_coef,
                               self.sld_floor,
                               grid.drr, grid.dth, m, w.ffr, w.ffth)
            artdif.sld_diffuse(ds_mr, self.vrr, self.jacV_r, self.jacV_th,
                               self.csp_r, self.csp_th, self.sld_coef,
                               self.sld_floor,
                               grid.drr, grid.dth, m, w.ffr, w.ffth)
            artdif.sld_diffuse(ds_mt, self.vth, self.jacV_r, self.jacV_th,
                               self.csp_r, self.csp_th, self.sld_coef,
                               self.sld_floor,
                               grid.drr, grid.dth, m, w.ffr, w.ffth)
            # 密度にも掛ける (音波の格子スケール振動を抑える)。保存量は
            # ∫ζ²ρ1 dV なので、連続の式と同じく発散に 1/ζ² を掛ける
            artdif.sld_diffuse_scaled(dq_ro, self.ro1, self.jacM_r,
                                      self.jacM_th, self.csp_r, self.csp_th,
                                      self.sld_coef, self.sld_floor,
                                      grid.drr, grid.dth, m,
                                      self.izeta2, w.ffr, w.ffth)

            # エントロピーにも掛ける。ここを忘れると s1 の格子ノイズが
            # 減衰せず、p1 = p0(gamma*rho1/rho0 + s1) を通して rho0 の小さい
            # 対流層上部で巨大な加速を生み、計算が壊れる
            artdif.sld_diffuse_primitive(dse1, self.se1, self.jacM_r,
                                         self.jacM_th, self.iJM,
                                         self.csp_r, self.csp_th,
                                         self.sld_coef, self.sld_floor,
                                         grid.drr, grid.dth, m, w.ffr, w.ffth)

        # --- エントロピー -------------------------------------------------
        hydro.entropy_rhs(dse1, self.se1, self.vrr, self.vth, s.ro0, s.tm0,
                          s.pr0, s.hp, s.delta, st.kappa_t, st.kappa_t_m,
                          s.JM, s.iJM, grid.rr, grid.sinTH, grid.sinTHm,
                          s.gamma, grid.drr, grid.dth, m, w.ffr, w.ffth)
        # 散逸したエネルギーをエントロピーに戻す (係数はちょうど 1)
        hydro.add_dissipative_heating(dse1, ds_mr, ds_mt, ds_om,
                                      self.vrr, self.vth, self.om1, cfg.om0,
                                      s.pr0, s.iJM, s.gamma, m)
        if self.magnetic:
            hydro.add_ohmic_heating(dse1, self.brr, self.bth, self.bph,
                                    st.et, s.pr0, grid.RR, grid.sinTH,
                                    s.gamma, grid.drr, grid.dth, m)

        # --- 散逸項を本体に合流 ------------------------------------------
        dq_mr += ds_mr
        dq_mt += ds_mt
        dq_om += ds_om
        return (dq_ro, dq_mr, dq_mt, dq_om, dse1)

    # -- 時間積分 ---------------------------------------------------------
    def step(self, dt):
        """SSP-RK2 (Heun) で 1 ステップ進める."""
        q0 = self.conserved()
        k1 = self.rhs()
        q1 = tuple(a + dt*b for a, b in zip(q0, k1))
        self.set_primitive_from_conserved(q1)
        k2 = self.rhs()
        qn = tuple(0.5*(a + b + dt*c) for a, b, c in zip(q0, q1, k2))
        self.set_primitive_from_conserved(qn)

    def cfl_dt(self):
        """CFL 条件から許容タイムステップを返す."""
        cfg, s, st = self.cfg, self.strat, self.setup
        kappa_max = max(float(np.max(st.nu_dif)), float(np.max(st.kappa_t)))
        if self.magnetic:
            kappa_max = max(kappa_max, float(np.max(st.et)))
        if self.use_artdif:
            # 人工拡散も陽解法なので安定条件に入れる
            self._update_characteristic_speed()
            kappa_max = max(kappa_max, artdif.sld_diffusivity_max(
                self.csp_r, self.csp_th, self.sld_coef, self.sld_floor,
                self.grid.drr, self.grid.dth, self.grid.rr, self.m))
        return hydro.cfl_dt(self.vrr, self.vth, self.brr, self.bth, self.bph,
                            s.ro0, self.ro1, s.cs_eff, self.grid.rr,
                            self.grid.drr, self.grid.dth, kappa_max, self.m,
                            getattr(cfg, 'cfl_safety', 0.2), self.magnetic)
