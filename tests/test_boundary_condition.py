"""境界条件のテスト(ゴースト充填の性質検証)。"""
import numpy as np
import pytest

import S2MFD
from S2MFD.physics import boundary_condition
from conftest import make_cfg, make_grid


def _random_fields(grid, seed=0):
    rng = np.random.default_rng(seed)
    Bph = rng.standard_normal((grid.ixg, grid.jxg))
    Aph = rng.standard_normal((grid.ixg, grid.jxg))
    return Bph, Aph


def _apply(cfg, grid, Bph, Aph):
    legendre = S2MFD.Legendre(grid)
    return boundary_condition(Bph.copy(), Aph.copy(), cfg, grid, legendre)


class TestVerticalBC:
    def test_ghost_cell_relations(self):
        cfg = make_cfg(ix=16, jx=16, boundary_condition_type='vertical')
        grid = make_grid(cfg)
        Bph, Aph = _random_fields(grid)
        Bph, Aph = _apply(cfg, grid, Bph, Aph)
        m = grid.margin
        cols = np.s_[m:grid.jxg - m]
        # 上端: Bph 反対称 (Bph=0)、d(r Aph)/dr = 0 (r*Aph が一定)
        assert np.allclose(Bph[-1, cols], -Bph[-2, cols])
        assert np.allclose(grid.rr[-1] * Aph[-1, cols], grid.rr[-2] * Aph[-2, cols])
        # 下端: Aph 反対称 (Aph=0)、d(r Bph)/dr = 0
        assert np.allclose(Aph[0, cols], -Aph[1, cols])
        assert np.allclose(grid.rr[0] * Bph[0, cols], grid.rr[1] * Bph[1, cols])

    def test_pole_antisymmetry(self):
        cfg = make_cfg(ix=16, jx=16, boundary_condition_type='vertical')
        grid = make_grid(cfg)
        Bph, Aph = _random_fields(grid)
        Bph, Aph = _apply(cfg, grid, Bph, Aph)
        m = grid.margin
        rows = np.s_[m:grid.ixg - m]
        assert np.allclose(Bph[rows, 0], -Bph[rows, 1])
        assert np.allclose(Aph[rows, 0], -Aph[rows, 1])
        assert np.allclose(Bph[rows, -1], -Bph[rows, -2])
        assert np.allclose(Aph[rows, -1], -Aph[rows, -2])


class TestPotentialBC:
    def test_top_bph_antisymmetric(self):
        cfg = make_cfg(ix=16, jx=32, boundary_condition_type='potential')
        grid = make_grid(cfg)
        Bph, Aph = _random_fields(grid)
        Bph, Aph = _apply(cfg, grid, Bph, Aph)
        m = grid.margin
        cols = np.s_[m:grid.jxg - m]
        assert np.allclose(Bph[-1, cols], -Bph[-2, cols])

    def test_linearity_in_aph(self):
        """外部ポテンシャル場接続は Aph について線形。"""
        cfg = make_cfg(ix=16, jx=32, boundary_condition_type='potential')
        grid = make_grid(cfg)
        Bph, Aph = _random_fields(grid)
        _, Aph1 = _apply(cfg, grid, Bph, Aph)
        _, Aph2 = _apply(cfg, grid, Bph, 2.0 * Aph)
        assert np.allclose(Aph2[-1, :], 2.0 * Aph1[-1, :])

    def test_low_order_mode_reconstruction(self):
        """n=1 モード (Aph ∝ P¹₁ = -sinθ) は上端ゴーストで
        (r_top/r_ghost)² 減衰の外挿になる。"""
        cfg = make_cfg(ix=16, jx=64, boundary_condition_type='potential')
        grid = make_grid(cfg)
        m = grid.margin
        Aph = np.zeros((grid.ixg, grid.jxg))
        # 表面値として P11 = -sinθ を置く
        Aph[grid.ixg - 2, m:grid.jxg - m] = -np.sin(grid.th[m:grid.jxg - m])
        Bph = np.zeros_like(Aph)
        Bph, Aph = _apply(cfg, grid, Bph, Aph)
        r_top = grid.rr[grid.ixg - m - 1]
        r_ghost = grid.rr[grid.ixg - 1]
        expected = -np.sin(grid.th[m:grid.jxg - m]) * (r_top / r_ghost)**2
        # 中点則射影の離散化誤差(特に極付近)があるため緩めの許容
        assert np.allclose(Aph[-1, m:grid.jxg - m], expected, rtol=5e-3, atol=1e-3)


class TestEquatorBC:
    """北半球のみを解くときの赤道境界条件 (Rempel 2006 §2.2)。

    > "The boundary condition is A = B_Phi = 0 at the pole and
    >  dA/dtheta = B_Phi = 0 at the equator, which selects the dipole symmetry
    >  for the solution"

    双極子は :math:`A_\\varphi\\propto\\sin\\theta` で赤道について**対称**、
    :math:`B_\\varphi` は**反対称**。したがって赤道側のゴーストは
    A が同符号 (+1)、B が逆符号 (-1) の鏡像になる。
    極と同じ扱い (どちらも -1) にすると A に誤った節を作ってしまう。
    """

    def _half(self, **over):
        cfg = make_cfg(ix=16, jx=16, boundary_condition_type='vertical',
                       thmin=0.0, thmax=np.pi / 2, **over)
        return cfg, make_grid(cfg)

    def test_equator_side_is_symmetric_for_aph(self):
        cfg, grid = self._half()
        Bph, Aph = _random_fields(grid)
        Bph, Aph = _apply(cfg, grid, Bph, Aph)
        m = grid.margin
        rows = np.s_[m:grid.ixg - m]
        # 極側 (j=0) は従来どおり両方とも反対称
        assert np.allclose(Bph[rows, 0], -Bph[rows, 1])
        assert np.allclose(Aph[rows, 0], -Aph[rows, 1])
        # 赤道側 (j=-1) は B_phi だけ反対称、A_phi は対称
        assert np.allclose(Bph[rows, -1], -Bph[rows, -2])
        assert np.allclose(Aph[rows, -1], +Aph[rows, -2])

    def test_full_sphere_keeps_pole_condition(self):
        """全球 [0, pi] のときは両端とも極なので従来どおり。"""
        cfg = make_cfg(ix=16, jx=16, boundary_condition_type='vertical')
        grid = make_grid(cfg)
        Bph, Aph = _random_fields(grid)
        Bph, Aph = _apply(cfg, grid, Bph, Aph)
        m = grid.margin
        rows = np.s_[m:grid.ixg - m]
        assert np.allclose(Aph[rows, -1], -Aph[rows, -2])

    def test_hydro_polar_bc_signs_are_valid_at_equator(self):
        """流体側の鏡像符号は極と赤道で一致するので変更不要 (記録)。

        赤道対称な流れでは ro1, v_r, Omega_1, s_1 が対称、v_theta が反対称。
        これは極での正則性条件と同じ符号なので ``apply_polar_bc`` を
        そのまま赤道に使える。
        """
        from S2MFD.physics.dynamic import _mirror_th
        m = 1
        jxg = 8
        for sign in (+1.0, -1.0):
            q = np.zeros((3, jxg))
            q[:, m:jxg - m] = np.arange(1, jxg - 2 * m + 1)
            _mirror_th(q, m, sign)
            assert np.allclose(q[:, -1], sign * q[:, -2])


class TestCornerGhosts:
    """4 隅のゴーストも埋めること。

    ``boundary_condition`` の緯度ループは長く
    ``rows = margin:ixg-margin`` に限定されていたため、**4 隅が一度も
    書かれず**古い値が残っていた (実測で 9 T、物理セルの最大は 1.5 T)。

    5 点ステンシルは角を参照しないので一見無害に見えるが、実測すると
    3 ステップ後の v_theta が相対 4.2e-4 変わる。ローレンツ力や人工拡散が
    動径ゴースト経由で角の情報を拾うため。

    流体側 (``_mirror_r`` / ``_mirror_th``) は全 i / 全 j を走るので
    角も埋まる。磁場側だけの問題だった。

    正しい角の値は「動径の鏡像符号 x 緯度の鏡像符号 x 対角の物理セル」。
    動径パスを先に (物理 j の範囲で) かけ、緯度パスを**全 i** でかければ
    自動的にそうなる。
    """

    def _apply(self, **over):
        cfg = make_cfg(ix=16, jx=16, boundary_condition_type='vertical', **over)
        grid = make_grid(cfg)
        Bph, Aph = _random_fields(grid)
        return cfg, grid, boundary_condition(Bph.copy(), Aph.copy(), cfg, grid, None)

    def test_corners_are_filled(self):
        """角が「動径符号 x 緯度符号 x 対角の物理セル」になること。"""
        cfg, grid, (Bph, Aph) = self._apply()
        m = grid.margin
        # 'vertical' の上端: Bph 反対称、下端: Aph 反対称。緯度は両端とも
        # 極なので Bph も Aph も反対称。したがって角は (-1)*(-1) = +1 倍。
        # 下端 x 極側の角 (i=0, j=0) は 物理セル (2m-1, 2m-1) の
        #   Aph: (-1)*(-1) = +1、Bph: (+r 比)*(-1)
        assert np.isclose(Aph[0, 0], Aph[2*m-1, 2*m-1]), \
            "下端x極 の角が埋まっていない"
        assert np.isclose(Aph[0, -1], Aph[2*m-1, grid.jxg-2*m]), \
            "下端x反対側 の角が埋まっていない"
        # 上端は d(r A)/dr = 0 なので r*A が保たれ (符号 +1)、極で -1 倍。
        assert np.isclose(Aph[-1, 0] * grid.rr[-1],
                          -Aph[grid.ixg-2*m, 2*m-1] * grid.rr[grid.ixg-2*m]), \
            "上端x極 の角が埋まっていない"

    def test_solution_does_not_depend_on_stale_corners(self):
        """角に何が入っていても、境界条件を通せば同じ結果になること。"""
        cfg = make_cfg(ix=16, jx=16, boundary_condition_type='vertical')
        grid = make_grid(cfg)
        m = grid.margin
        base_B, base_A = _random_fields(grid)
        out = []
        for poison in (0.0, 1.0e6):
            B, A = base_B.copy(), base_A.copy()
            for a in (B, A):
                a[:m, :m] = poison; a[:m, -m:] = poison
                a[-m:, :m] = poison; a[-m:, -m:] = poison
            out.append(boundary_condition(B, A, cfg, grid, None))
        assert np.array_equal(out[0][0], out[1][0]), "Bph が角の初期値に依存する"
        assert np.array_equal(out[0][1], out[1][1]), "Aph が角の初期値に依存する"


class TestBCImplementationsAgree:
    """磁場の境界条件は 2 箇所に重複実装されている。食い違わせないこと。

    * ``S2MFD/physics/boundary_condition.py`` — Python 版。DynamicSolver と
      Rempel 2006 のドライバ (dynamo7.py) が使う。
    * ``S2MFD/physics/stepping.py`` の njit ``bc`` — numba バッチ経路
      ``advance_to`` 用。既存の運動学的ダイナモが使う。

    2026-08-23 に前者だけ角ゴーストの修正を入れたところ、
    test_kernel_equivalence の「バッチと手動ループが一致すること」が
    **角セルだけで**落ちた (物理セルは厳密一致)。どちらかを触ったら
    必ずもう一方も直すこと。このテストはその見張り。
    """

    @pytest.mark.parametrize('bc_type', ['vertical', 'potential'])
    def test_two_implementations_give_same_ghosts(self, bc_type):
        from S2MFD.physics.stepping import _make_bc, BC_CODE
        cfg = make_cfg(ix=16, jx=16, boundary_condition_type=bc_type)
        grid = make_grid(cfg)
        legendre = S2MFD.Legendre(grid) if bc_type == 'potential' else None
        B0, A0 = _random_fields(grid, seed=3)

        Bp, Ap = boundary_condition(B0.copy(), A0.copy(), cfg, grid, legendre)

        bc = _make_bc(BC_CODE[bc_type])
        op = (legendre.potential_operator if legendre is not None
              else np.zeros((grid.margin, 1, 1)))
        Bn, An = bc(B0.copy(), A0.copy(), grid.rr, grid.margin, op)

        assert np.allclose(Bp, Bn, rtol=1e-12, atol=0), \
            f"{bc_type}: Bph のゴーストが 2 実装で違う"
        assert np.allclose(Ap, An, rtol=1e-12, atol=0), \
            f"{bc_type}: Aph のゴーストが 2 実装で違う"
