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
