"""背景成層 (Rempel 2005 のポリトロープ) と保存形のヤコビアン.

運動学的 (kinematic) ダイナモでは背景の熱力学構造を必要としないため,
このクラスは ``cfg.dynamics != 'kinematic'`` のときだけ構築される.

背景構造
--------
Rempel (2005) に従い, 対流層を断熱ポリトロープで近似する. 対流層底
:math:`r_{\\rm bc}` での値 :math:`(\\rho_{\\rm bc}, p_{\\rm bc},
T_{\\rm bc}, g_{\\rm bc})` を与え, 圧力スケールハイト
:math:`H_{\\rm bc} = p_{\\rm bc}/(\\rho_{\\rm bc} g_{\\rm bc})` を使って

.. math::
   \\xi(r) &= 1 + \\frac{\\gamma-1}{\\gamma}
              \\frac{r_{\\rm bc}}{H_{\\rm bc}}
              \\left(\\frac{r_{\\rm bc}}{r} - 1\\right) \\\\
   \\rho_0 &= \\rho_{\\rm bc}\\,\\xi^{1/(\\gamma-1)}, \\quad
   p_0 = p_{\\rm bc}\\,\\xi^{\\gamma/(\\gamma-1)}, \\quad
   T_0 = T_{\\rm bc}\\,\\xi, \\quad
   g = g_{\\rm bc}\\left(\\frac{r_{\\rm bc}}{r}\\right)^2

とする. :math:`\\xi` は :math:`r` の増加とともに減少し, ある半径で 0 に
達する (太陽パラメタでは :math:`r \\simeq 1.003\\,R_\\odot`). そこを超えると
密度が非物理的になるため, 上部境界はそれより内側に取る必要がある
(Rempel 2006 / 齋藤 (2024) はいずれも :math:`0.96\\,R_\\odot`).

状態方程式と摂動
----------------
理想気体 :math:`p = \\rho (R/\\mu) T` を仮定し, 線形化した圧力摂動を

.. math::
   p_1 = p_0\\left(\\gamma\\frac{\\rho_1}{\\rho_0} + \\frac{s_1}{c_v}\\right)

で与える (:math:`s = c_v\\ln(p\\rho^{-\\gamma})` の線形化).

音速抑制法 (RSST)
-----------------
連続の式を :math:`\\partial\\rho_1/\\partial t = -\\xi_s^{-2}\\nabla\\cdot
(\\rho_0\\boldsymbol{v})` と修正して音波の位相速度を :math:`c_s/\\xi_s` に
落とす (Hotta, Rempel & Yokoyama 2012). 保存量は
:math:`\\int(\\rho_0 + \\xi_s^2\\rho_1)\\,dV` となる.

保存形のヤコビアン
------------------
:mod:`S2MFD.physics.conservative` の規約に合わせ, セル体積のヤコビアンを
吸収した係数をあらかじめ 2 次元配列として持つ. これにより時間積分カーネル
から ``sin``/``**`` などの高コスト演算が消える.
"""
import numpy as np

from S2MFD.npz_io import NpzIO

__all__ = ['Stratification']


class Stratification(NpzIO):
    """背景成層と保存形の幾何係数.

    Attributes
    ----------
    ro0, pr0, tm0, gr, hp : numpy.ndarray
        密度・圧力・温度・重力加速度・圧力スケールハイトの 1 次元動径分布
        (``ixg``,). CGS 単位.
    cs0 : numpy.ndarray
        断熱音速 :math:`\\sqrt{\\gamma p_0/\\rho_0}` (``ixg``,).
    cs_eff : numpy.ndarray
        RSST で抑制された実効音速 :math:`c_s/\\xi_s` (``ixg``,).
    zeta : numpy.ndarray
        音速抑制係数 :math:`\\xi_s(r)` (``ixg``,). ``cfg.rsst_type`` により
        一定値または動径依存.
    cp, cv, rgas : float
        定圧比熱・定積比熱・気体定数 :math:`R/\\mu`.
    JM : numpy.ndarray
        :math:`r^2\\sin\\theta` — 質量・エントロピーの保存量ヤコビアン.
    JV : numpy.ndarray
        :math:`r^2\\sin\\theta\\,\\rho_0` — 運動量の保存量ヤコビアン.
    JL : numpy.ndarray
        :math:`r^4\\sin^3\\theta\\,\\rho_0` — 角運動量の保存量ヤコビアン.
        r 方向フラックスの係数も同じ.
    JLY : numpy.ndarray
        :math:`r^3\\sin^3\\theta\\,\\rho_0` — 角運動量の theta 方向
        フラックス係数 (r の冪が 1 つ低い).
    JVY : numpy.ndarray
        :math:`r\\sin\\theta\\,\\rho_0` — 質量・運動量の theta 方向
        フラックス係数.
    """

    def __init__(self, cfg, grid):
        """背景成層を構築する.

        Parameters
        ----------
        cfg : S2MFD.Cfg
            ``gamma``, ``rr_bc``, ``ro_bc``, ``pr_bc``, ``tm_bc``,
            ``gr_bc``, ``rsst_zeta`` などを参照する.
        grid : S2MFD.Grid
            格子オブジェクト.
        """
        gm = cfg.gamma
        self.gamma = gm

        # --- 対流層底での参照値からポリトロープを組む -------------------
        hp_bc = cfg.pr_bc/(cfg.ro_bc*cfg.gr_bc)
        xi = 1.0 + (gm - 1.0)/gm*cfg.rr_bc/hp_bc*(cfg.rr_bc/grid.rr - 1.0)

        # 物理セル内でポリトロープが破綻していないか確認する.
        # ゴーストセルは外挿なので判定から外す.
        i0, i1 = grid.margin, grid.ixg - grid.margin
        if np.any(xi[i0:i1] <= 0.0):
            rr_zero = cfg.rr_bc/(1.0 - gm/(gm - 1.0)*hp_bc/cfg.rr_bc)
            raise ValueError(
                'Rempel(2005) のポリトロープが計算領域内で密度ゼロに達する: '
                f'rrmax={grid.rrmax:.4e} cm ({grid.rrmax/cfg.RSUN:.4f} RSUN) だが '
                f'ポリトロープの外端は {rr_zero:.4e} cm '
                f'({rr_zero/cfg.RSUN:.4f} RSUN). rrmax をこれより内側に取ること '
                '(Rempel 2006 / 齋藤 2024 はいずれも 0.96 RSUN).')
        # ゴーストセル側だけは床を張って NaN を防ぐ (境界条件で上書きされる).
        xi = np.maximum(xi, 1.0e-12)

        self.xi_poly = xi
        self.ro0 = cfg.ro_bc*xi**(1.0/(gm - 1.0))
        self.pr0 = cfg.pr_bc*xi**(gm/(gm - 1.0))
        self.tm0 = cfg.tm_bc*xi
        self.gr = cfg.gr_bc*(cfg.rr_bc/grid.rr)**2
        self.hp = self.pr0/(self.ro0*self.gr)

        # r 面上の rho0 (面 i はセル i-1 と i の境界). 拡散フラックスの係数に
        # 使う。両隣のセルで同一の値を使うことが telescoping の前提。
        self.ro0m = np.zeros_like(self.ro0)
        self.ro0m[1:] = 0.5*(self.ro0[1:] + self.ro0[:-1])

        # --- 熱力学定数 --------------------------------------------------
        self.rgas = cfg.pr_bc/(cfg.ro_bc*cfg.tm_bc)   # R/mu
        self.cv = self.rgas/(gm - 1.0)
        self.cp = gm*self.cv

        # --- 超断熱度 delta(r) と背景エントロピー勾配 --------------------
        self.delta = self._build_delta(cfg, grid)
        # Rempel 2005 式 (8): ds0/dr = -gamma*delta/Hp
        # エントロピー方程式にはこの符号を反転した +v_r*gamma*delta/Hp が入る。
        self.dsdr0 = -gm*self.delta/self.hp

        # --- 音速と RSST ------------------------------------------------
        self.cs0 = np.sqrt(gm*self.pr0/self.ro0)
        self.zeta = self._build_zeta(cfg, grid)
        self.cs_eff = self.cs0/self.zeta

        # --- 保存形のヤコビアン (2 次元, 事前計算) ----------------------
        RO0 = self.ro0[:, None]
        sin1 = grid.sinTH
        sin3 = sin1**3
        self.JM = grid.RR**2*sin1                 # 質量・エントロピー
        self.JV = self.JM*RO0                     # 運動量 (r 方向フラックスも同じ)
        self.JVY = grid.RR*sin1*RO0               # 運動量の theta フラックス
        self.JL = grid.RR**4*sin3*RO0             # 角運動量 (r フラックスも同じ)
        self.JLY = grid.RR**3*sin3*RO0            # 角運動量の theta フラックス
        self.RSIN = grid.RR*sin1                  # r sin(theta) (円筒半径)
        self.W2 = self.RSIN**2                    # varpi^2 (比角運動量の係数)

        # 保存量からプリミティブ変数へ戻すための逆数.
        # 極のゴーストセルでは sin(theta) < 0 になりうるので, 物理セルの外は
        # ゼロにして誤用を早期に検出できるようにする.
        self.iJM = _safe_reciprocal(self.JM, grid)
        self.iJV = _safe_reciprocal(self.JV, grid)
        self.iJL = _safe_reciprocal(self.JL, grid)

    def _build_delta(self, cfg, grid):
        """超断熱度 :math:`\\delta = \\nabla - \\nabla_{\\rm ad}` (Rempel 2005 式 25-26).

        .. math::
           \\delta &= \\delta_{\\rm conv}
             + \\tfrac12(\\delta_{\\rm os}-\\delta_{\\rm conv})
               \\left[1-\\tanh\\frac{r-r_{\\rm tran}}{d_{\\rm tran}}\\right] \\\\
           \\delta_{\\rm conv} &= \\delta_{\\rm top}
               e^{(r-r_{\\max})/d_{\\rm top}}
             + \\delta_{\\rm cz}\\frac{r-r_{\\rm sub}}{r_{\\max}-r_{\\rm sub}}

        :math:`\\delta>0` が超断熱 (対流不安定), :math:`\\delta<0` が亜断熱.
        Rempel 2006 の参照モデル (= 2005 の case 1) は対流層が断熱
        (:math:`\\delta_{\\rm conv}=0`) で, オーバーシュート層のみ
        :math:`\\delta_{\\rm os}=-1.5\\times10^{-5}` の亜断熱になる.

        音速抑制法では :math:`\\delta` が :math:`\\xi_s^2` 倍にスケールされる
        (Rempel 2005 式 34) ため, 太陽の放射層の実際の値
        :math:`\\delta\\sim-0.1` は表現できない. オーバーシュート程度の
        :math:`10^{-5}` なら問題ない, と原論文が明記している.
        """
        rmax = grid.rrmax
        d_top = getattr(cfg, 'd_top', 0.0125*cfg.RSUN)
        d_conv = (getattr(cfg, 'delta_top', 0.0)*np.exp((grid.rr - rmax)/d_top)
                  + getattr(cfg, 'delta_cz', 0.0)
                  * (grid.rr - cfg.r_sub)/(rmax - cfg.r_sub))
        return d_conv + 0.5*(cfg.delta_os - d_conv)*(
            1.0 - np.tanh((grid.rr - cfg.r_tran)/cfg.d_tran))

    def _build_zeta(self, cfg, grid):
        """音速抑制係数 :math:`\\xi_s(r)` を作る.

        ``'const'``
            一定値 ``cfg.rsst_zeta`` (齋藤 (2024) と同じ. 既定).
        ``'mach'``
            実効音速を :math:`c_{s,\\rm eff} = v_{\\rm ref}/{\\rm Ma}`
            で動径方向に一定にする. すなわち :math:`\\xi_s(r)\\propto c_s(r)`.

        なぜ「実効音速の一様化」だけでは速くならないか
        ----------------------------------------------
        CFL は :math:`\\Delta t \\propto \\min_r \\Delta l(r)/c_{s,\\rm eff}(r)`
        で決まる. 本モデルの格子は :math:`\\Delta r` が一様で,
        しかも :math:`\\Delta r \\ll r\\Delta\\theta` なので
        :math:`\\Delta l` は動径方向にほぼ一定である. したがって
        「:math:`c_{s,\\rm eff}` を最も厳しい点の値に揃える」ような規格化を
        しても :math:`\\Delta t` は 1 ミリも増えない.

        効くのは**規格化の基準を精度側に置く**ことである. RSST の誤差は
        実効マッハ数の 2 乗で効く (Hotta, Rempel & Yokoyama 2012) ので,
        許容マッハ数 ``cfg.rsst_mach`` を決めれば実効音速の上限が決まり,
        そこから :math:`\\xi_s` が決まる. 深部ほど :math:`c_s` が大きいので
        :math:`\\xi_s` も大きくなり, 一定 :math:`\\xi_s` より大きな
        :math:`\\Delta t` を取れる.
        """
        kind = getattr(cfg, 'rsst_type', 'const')
        if kind == 'const':
            return np.full(grid.ixg, float(cfg.rsst_zeta))
        if kind == 'mach':
            v_ref = getattr(cfg, 'rsst_vref', 2.0e3)   # 代表流速 [cm/s]
            mach = getattr(cfg, 'rsst_mach', 0.05)
            cs_target = v_ref/mach
            # xi_s < 1 は音速を「速める」ことになるので 1 で床を張る
            return np.maximum(self.cs0/cs_target, 1.0)
        raise ValueError(f'unknown rsst_type: {kind!r}')

    def pressure_perturbation(self, ro1, se1):
        """線形化した圧力摂動 :math:`p_1` を返す (Rempel 2005 式 6).

        .. math:: p_1 = p_0\\left(\\gamma\\frac{\\rho_1}{\\rho_0} + s_1\\right)

        :math:`s_1` は **:math:`c_v` で規格化された無次元エントロピー**
        (:math:`s=\\ln(p\\rho^{-\\gamma})`). :math:`c_p` ではないことに注意.
        物理エントロピーとの関係は :math:`s_{\\rm phys}=c_v s_1`.
        齋藤 (2024) コードの ``en1`` と同一の量.
        """
        return self.pr0[:, None]*(self.gamma*ro1/self.ro0[:, None] + se1)


def _safe_reciprocal(jac, grid):
    """物理セルでは ``1/jac``, ゴーストセルでは 0 を返す.

    極のゴーストセルでは :math:`\\sin\\theta<0` となり ``JL`` の符号が
    反転する. そこを保存量から復元しようとすると符号が壊れるので,
    そもそも計算しないよう 0 を入れて壊れ方を目立たせる.
    """
    out = np.zeros_like(jac)
    i0, i1 = grid.margin, grid.ixg - grid.margin
    j0, j1 = grid.margin, grid.jxg - grid.margin
    out[i0:i1, j0:j1] = 1.0/jac[i0:i1, j0:j1]
    return out
