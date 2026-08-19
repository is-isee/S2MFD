"""エネルギー収支の診断 (Rempel 2006 式 20-34).

質量と角運動量は machine precision で保存させられるが, **全エネルギーは
保存しない**. 音速抑制法 (連続の式だけを変形するのでエネルギー方程式と
整合しない), 乱流拡散, :math:`\\Lambda` 効果 (対流エネルギー流束から
差動回転にエネルギーを渡す寄生的な項) がいずれもエネルギーを出し入れ
するためである.

そこで「保存させる」代わりに**収支を項ごとに追跡できるようにする**.
Rempel (2006) が表 1 で報告している各項と直接比較できる形にしてある.

エネルギー貯留 (式 23-25)
--------------------------
.. math::
   E_\\Omega &= \\int dV\\,\\tfrac12\\rho_0\\varpi^2\\Omega^2 \\\\
   E_M &= \\int dV\\,\\tfrac12\\rho_0 v_m^2 \\\\
   E_B &= \\int dV\\,\\frac{B_\\Phi^2}{8\\pi}

:math:`\\varpi=r\\sin\\theta`, :math:`\\Omega=\\Omega_0+\\Omega_1`,
:math:`v_m^2=v_r^2+v_\\theta^2`. 原論文は SI なので :math:`B^2/(2\\mu_0)`,
本実装は CGS なので :math:`B^2/(8\\pi)`.

収支 (式 20-22)
----------------
.. math::
   \\partial_t E_\\Omega &= Q_\\Lambda - Q_\\nu^\\Omega - Q_C - Q_L^\\Omega \\\\
   \\partial_t E_M &= Q_C - Q_\\nu^M - Q_B - Q_L^M \\\\
   \\partial_t E_B &= Q_L^\\Omega + Q_L^M - Q_\\eta

定常状態では左辺がゼロになるので

.. math::
   Q_\\Lambda = Q_\\nu^\\Omega + Q_C + Q_L^\\Omega,\\quad
   Q_C = Q_\\nu^M + Q_B + Q_L^M,\\quad
   Q_\\eta = Q_L^\\Omega + Q_L^M

が成り立つはずである. 原論文は「エネルギー交換項の精度は 0.001 程度なので
これらの関係はその誤差の範囲でしか満たされない」と注記しており,
:func:`budget_residuals` はこの残差を返すので直接比較できる.

原論文の疑わしい記述への対応
----------------------------
式 (33) の :math:`Q_\\eta` には :math:`\\mu_0` が印字されていないが,
次元が合わないため補っている (CGS なので :math:`1/4\\pi`). 式 (25) の
:math:`E_B` には :math:`\\mu_0` があることから印刷上の脱落と判断した.
詳細は ``doc/dev_records/2026-08-19_rempel_equations.md`` §11.
"""
import numpy as np

__all__ = ['EnergyBudget', 'solar_luminosity']

#: 太陽光度 [erg/s]. Rempel は各項を :math:`F_\\odot` 単位で報告している.
solar_luminosity = 3.828e33

FOUR_PI = 4.0*np.pi
EIGHT_PI = 8.0*np.pi


class EnergyBudget:
    """Rempel (2006) 式 (20)-(34) のエネルギー収支診断.

    Parameters
    ----------
    grid : S2MFD.Grid
    strat : S2MFD.stratification.Stratification
    setup : S2MFD.Setup
    cfg : S2MFD.Cfg
    """

    def __init__(self, cfg, grid, strat, setup):
        self.cfg, self.grid, self.strat, self.setup = cfg, grid, strat, setup
        m = grid.margin
        self.sl = (slice(m, grid.ixg - m), slice(m, grid.jxg - m))
        # 体積要素. 全球 [0, pi] を解くので方位角の 2pi を掛ける
        # (Rempel は北半球のみなので 4pi を掛けている. 対称解なら同じ値).
        self.dV = 2.0*np.pi*grid.RR**2*grid.sinTH*grid.drr*grid.dth
        self.varpi = strat.RSIN

    # -- 微分 (中心差分) --------------------------------------------------
    def _ddr(self, qq):
        out = np.zeros_like(qq)
        out[1:-1] = (qq[2:] - qq[:-2])/(2.0*self.grid.drr)
        return out

    def _ddth(self, qq):
        out = np.zeros_like(qq)
        out[:, 1:-1] = (qq[:, 2:] - qq[:, :-2])/(2.0*self.grid.dth)
        return out

    def _integrate(self, qq):
        return float((qq*self.dV)[self.sl].sum())

    # -- 貯留 (式 23-25) --------------------------------------------------
    def reservoirs(self, om1, vrr, vth, bph):
        """:math:`(E_\\Omega, E_M, E_B)` を返す [erg]."""
        s = self.strat
        om = self.cfg.om0 + om1
        e_om = self._integrate(0.5*s.ro0[:, None]*self.varpi**2*om**2)
        e_m = self._integrate(0.5*s.ro0[:, None]*(vrr**2 + vth**2))
        e_b = self._integrate(bph**2/EIGHT_PI)
        return e_om, e_m, e_b

    def differential_rotation_energy(self, om1):
        """剛体回転を除いた差動回転のエネルギー.

        :math:`E_\\Omega` は :math:`\\Omega_0` が支配的で変化が埋もれるため,
        トーショナル振動などを見るときはこちらを使う.
        """
        s = self.strat
        return self._integrate(0.5*s.ro0[:, None]*self.varpi**2*om1**2)

    # -- 交換項 (式 26-33) ------------------------------------------------
    def exchanges(self, om1, vrr, vth, brr, bth, bph, se1):
        """各エネルギー交換項を dict で返す [erg/s].

        Returns
        -------
        dict
            ``Q_Lambda``, ``Q_nu_Omega``, ``Q_C``, ``Q_L_Omega``,
            ``Q_nu_M``, ``Q_B``, ``Q_L_M``, ``Q_eta``
        """
        cfg, grid, s, st = self.cfg, self.grid, self.strat, self.setup
        rr = grid.rr[:, None]
        ro0 = s.ro0[:, None]
        nu_lam = st.nu_lam[:, None]
        nu_dif = st.nu_dif[:, None]
        w = self.varpi
        om = cfg.om0 + om1

        dom_dr = self._ddr(om)
        dom_dt = self._ddth(om)/rr

        # 式 (26): Lambda 効果が差動回転に注入するエネルギー
        q_lambda = -self._integrate(
            nu_lam*ro0*w*(dom_dr*st.lam_rp + dom_dt*st.lam_tp))

        # 式 (27): 差動回転の粘性散逸
        q_nu_om = self._integrate(
            nu_dif*ro0*w**2*(dom_dr**2 + dom_dt**2))

        # 式 (28): 差動回転 -> 子午面循環 (遠心力を通じた変換)
        half_om2 = 0.5*om**2
        q_c = -self._integrate(
            w**2*ro0*(vrr*self._ddr(half_om2) + vth*self._ddth(half_om2)/rr))

        # 式 (30): 子午面循環の粘性散逸
        q_nu_m = self._viscous_meridional_dissipation(vrr, vth)

        # 式 (31): 浮力による仕事 (子午面循環 <-> 内部エネルギー)
        q_b = -self._integrate(vrr*ro0*s.gr[:, None]*se1/s.gamma)

        # 式 (29), (32), (33): 磁場との交換
        wb = w*bph
        dwb_dr = self._ddr(wb)
        dwb_dt = self._ddth(wb)/rr
        q_l_om = -self._integrate(om*(brr*dwb_dr + bth*dwb_dt)/FOUR_PI)
        q_l_m = self._integrate(
            bph/np.where(w != 0.0, w, np.inf)
            * (vrr*dwb_dr + vth*dwb_dt)/FOUR_PI)
        w2 = np.where(w != 0.0, w**2, np.inf)
        q_eta = self._integrate(
            st.et/w2*(dwb_dr**2 + dwb_dt**2)/FOUR_PI)

        return {'Q_Lambda': q_lambda, 'Q_nu_Omega': q_nu_om, 'Q_C': q_c,
                'Q_L_Omega': q_l_om, 'Q_nu_M': q_nu_m, 'Q_B': q_b,
                'Q_L_M': q_l_m, 'Q_eta': q_eta}

    def _viscous_meridional_dissipation(self, vrr, vth):
        """式 (30): :math:`\\tfrac12\\sum R_{ik}E_{ik}` (子午面成分).

        変形テンソルは Rempel 2005 式 (13)-(16), 応力は式 (12) の
        :math:`R_{ik}=\\nu_t\\rho_0(E_{ik}-\\tfrac23\\delta_{ik}\\nabla\\cdot v)`.
        """
        grid, s, st = self.grid, self.strat, self.setup
        rr = grid.rr[:, None]
        ro0 = s.ro0[:, None]
        nu = st.nu_dif[:, None]
        cot = grid.cosTH/np.where(grid.sinTH != 0.0, grid.sinTH, np.inf)

        e_rr = 2.0*self._ddr(vrr)
        e_tt = 2.0*self._ddth(vth)/rr + 2.0*vrr/rr
        e_pp = 2.0*(vrr + vth*cot)/rr
        e_rt = rr*self._ddr(vth/rr) + self._ddth(vrr)/rr
        div_v = 0.5*(e_rr + e_tt + e_pp)   # E の対角和の半分 = div v

        def stress(eik, diag):
            return nu*ro0*(eik - (2.0/3.0)*div_v*(1.0 if diag else 0.0))

        return self._integrate(0.5*(
            stress(e_rr, True)*e_rr + 2.0*stress(e_rt, False)*e_rt
            + stress(e_tt, True)*e_tt + stress(e_pp, True)*e_pp))

    # -- 収支の残差 --------------------------------------------------------
    @staticmethod
    def budget_residuals(q, de_om_dt=0.0, de_m_dt=0.0, de_b_dt=0.0):
        """収支式 (20)-(22) の残差を返す.

        定常状態では 3 つとも 0 になるはず. 原論文は精度 0.001 程度と
        注記しているので, :math:`Q_\\Lambda` で規格化した値がその程度に
        収まっていれば期待どおり.

        Returns
        -------
        dict
            ``E_Omega``, ``E_M``, ``E_B`` の各収支式の残差 [erg/s] と,
            ``relative`` (:math:`Q_\\Lambda` で規格化した最大残差).
        """
        r_om = (q['Q_Lambda'] - q['Q_nu_Omega'] - q['Q_C'] - q['Q_L_Omega']
                - de_om_dt)
        r_m = q['Q_C'] - q['Q_nu_M'] - q['Q_B'] - q['Q_L_M'] - de_m_dt
        r_b = q['Q_L_Omega'] + q['Q_L_M'] - q['Q_eta'] - de_b_dt
        scale = abs(q['Q_Lambda']) if q['Q_Lambda'] != 0.0 else 1.0
        return {'E_Omega': r_om, 'E_M': r_m, 'E_B': r_b,
                'relative': max(abs(r_om), abs(r_m), abs(r_b))/scale}

    def report(self, om1, vrr, vth, brr, bth, bph, se1):
        """人が読める形の収支レポートを文字列で返す."""
        e_om, e_m, e_b = self.reservoirs(om1, vrr, vth, bph)
        q = self.exchanges(om1, vrr, vth, brr, bth, bph, se1)
        res = self.budget_residuals(q)
        lines = [
            f"E_Omega = {e_om:.4e} erg   (差動回転のみ "
            f"{self.differential_rotation_energy(om1):.4e})",
            f"E_M     = {e_m:.4e} erg",
            f"E_B     = {e_b:.4e} erg",
            f"Q_Lambda = {q['Q_Lambda']:.4e} erg/s "
            f"= {q['Q_Lambda']/solar_luminosity:.4f} L_sun",
        ]
        scale = abs(q['Q_Lambda']) if q['Q_Lambda'] != 0.0 else 1.0
        for k in ('Q_nu_Omega', 'Q_C', 'Q_L_Omega', 'Q_nu_M', 'Q_B',
                  'Q_L_M', 'Q_eta'):
            lines.append(f"{k:<11}= {q[k]:+.4e} erg/s  "
                         f"({q[k]/scale:+.4f} x Q_Lambda)")
        lines.append(f"収支の残差 (Q_Lambda 規格化, 定常なら ~0.001): "
                     f"{res['relative']:.4f}")
        return "\n".join(lines)
