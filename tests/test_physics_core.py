"""physics_core の各項のテスト(解析解との比較)。"""
import numpy as np
import pytest

import S2MFD
from S2MFD.physics import poloidal_mag, advection, alpha_effect
from conftest import make_cfg, make_grid


def _grid(ix=64, jx=64):
    cfg = make_cfg(ix=ix, jx=jx)
    return cfg, make_grid(cfg)


class TestPoloidalMag:
    def test_dipole_field(self):
        """A = sinθ/r² (双極子) → Brr = 2cosθ/r³, Bth = sinθ/r³。"""
        cfg, grid = _grid()
        # 数値安定のため r を RSUN 単位で扱う
        RR = grid.RR / cfg.RSUN
        Aph = grid.sinTH / RR**2
        Brr, Bth = poloidal_mag(Aph, RR, grid.sinTH,
                                grid.drr2 / cfg.RSUN, grid.dth)
        interior = np.s_[2:-2, 2:-2]
        assert np.allclose(Brr[interior], (2 * grid.cosTH / RR**3)[interior], rtol=5e-3)
        assert np.allclose(Bth[interior], (grid.sinTH / RR**3)[interior], rtol=5e-3)

    def test_zero_potential_gives_zero_field(self):
        cfg, grid = _grid(ix=16, jx=16)
        Aph = np.zeros((grid.ixg, grid.jxg))
        Brr, Bth = poloidal_mag(Aph, grid.RR, grid.sinTH, grid.drr2, grid.dth)
        assert np.all(Brr == 0)
        assert np.all(Bth == 0)


class TestAdvection:
    def test_flux_form_uniform_field_uniform_flow(self):
        """一様 B・一様 u_r で Bph_adrr = -B u_r d(r)/dr / r = -B u_r / r。"""
        cfg, grid = _grid(ix=32, jx=16)
        B0, u0 = 2.0, 3.0
        Bph = np.full((grid.ixg, grid.jxg), B0)
        Aph = np.zeros_like(Bph)
        urr = np.full_like(Bph, u0)
        uth = np.zeros_like(Bph)
        Bph_adrr, Bph_adth, Aph_adrr, Aph_adth = advection(
            Bph, Aph, grid.RR, grid.sinTH, urr, uth, grid.drr2, grid.dth)
        interior = np.s_[1:-1, :]
        # drr2(B u r) = B u (r は線形なので中心差分は厳密)
        assert np.allclose(Bph_adrr[interior], (-B0 * u0 / grid.RR)[interior])
        assert np.allclose(Bph_adth, 0.0)
        assert np.allclose(Aph_adrr, 0.0)
        assert np.allclose(Aph_adth, 0.0)


class TestAlphaEffect:
    def test_normal_type_quenching(self):
        cfg, grid = _grid(ix=8, jx=8)
        Bph = np.linspace(-3, 3, grid.ixg * grid.jxg).reshape(grid.ixg, grid.jxg)
        so = np.full_like(Bph, 1.7)
        out = alpha_effect(Bph, None, grid.rr, 3, so, 'normal')
        assert np.allclose(out, 1.7 * Bph / (1 + Bph**2))

    def test_bl_type_is_nonlocal(self):
        """BL 型はタコクライン (ibase) の B を全動径に放送する。"""
        cfg, grid = _grid(ix=8, jx=8)
        Bph = np.zeros((grid.ixg, grid.jxg))
        ibase = 3
        Bph[ibase, :] = 2.0
        so = np.ones_like(Bph)
        out = alpha_effect(Bph, None, grid.rr, ibase, so, 'BL')
        expected_row = 2.0 / (1 + 4.0)
        assert np.allclose(out, expected_row)
