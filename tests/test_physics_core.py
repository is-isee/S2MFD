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


class TestR06AlphaQuenching:
    """Rempel (2006) の α クエンチングの扱い。

    論文 §2.2 は運動学的参照解 (図 3) について
    "include α quenching with a quenching field strength of 1 T (10 kG)"
    と書く一方、§3.1 はローレンツ力フィードバックを入れた解について
    "Since Lorentz force feedback introduces enough nonlinearity to saturate
    the dynamo, it is not necessary to include α quenching" と明記し、
    図 4 のキャプションも "no α quenching" としている。
    したがって ``cfg.alpha_quenching`` で切り替えられる必要がある。
    """

    def _setup(self, quenching, amp):
        """1 ステップ進めて物理セルの (Bph, Aph) を返す。

        ゴーストセルは ``time_marching`` の管轄外 (呼び出し側が直後に
        ``boundary_condition`` で埋める) なので比較から除く。
        """
        from S2MFD.stratification import Stratification
        from S2MFD.physics import time_marching
        cfg = make_cfg('parameters/rempel06.py', ix=32, jx=32,
                       alpha_quenching=quenching)
        grid = make_grid(cfg)
        Stratification(cfg, grid)
        setup = S2MFD.Setup(cfg, grid)
        # 種磁場は 0.735 RSUN 付近に置く (h(r) カーネルが拾う範囲)
        prof = np.exp(-((grid.RR - 0.735 * cfg.RSUN) / (0.02 * cfg.RSUN)) ** 2)
        Bph = np.ascontiguousarray(amp * prof * np.sin(2 * grid.TH))
        Aph = np.zeros_like(Bph)
        m = grid.margin
        sl = (slice(m, grid.ixg - m), slice(m, grid.jxg - m))
        Bm, Am = time_marching(Bph, Aph, 1.0e3, cfg, grid, setup)
        return Bm[sl], Am[sl]

    def test_unquenched_source_is_linear_in_field(self):
        """クエンチングを外すと誘導方程式は磁場振幅について厳密に線形。

        Aph = 0 から始めれば α ソース以外の項はすべて (Bph, Aph) について
        線形なので、種磁場を 2 倍すれば 1 ステップ後も厳密に 2 倍になる。
        """
        B1, A1 = self._setup(False, 1.0e3)
        B2, A2 = self._setup(False, 2.0e3)
        assert np.abs(A1).max() > 0, "α ソースが立っていない"
        assert np.array_equal(A2, 2.0 * A1)
        assert np.array_equal(B2, 2.0 * B1)

    def test_quenched_source_saturates(self):
        """B_eq = 1 T (1e4 G) 前後でクエンチングが効き、線形性が崩れる。"""
        B1, A1 = self._setup(True, 1.0e4)
        B2, A2 = self._setup(True, 2.0e4)
        assert np.abs(A1).max() > 0
        # 2 倍にしても 2 倍にはならない (飽和する)
        assert np.abs(A2).max() < 1.8 * np.abs(A1).max()

    def test_default_is_quenched_for_backward_compatibility(self):
        """cfg に指定がなければ従来どおりクエンチングあり。"""
        from S2MFD.stratification import Stratification
        from S2MFD.physics import time_marching
        cfg = make_cfg('parameters/rempel06.py', ix=32, jx=32)
        if hasattr(cfg, 'alpha_quenching'):
            del cfg.alpha_quenching
        grid = make_grid(cfg)
        Stratification(cfg, grid)
        setup = S2MFD.Setup(cfg, grid)
        prof = np.exp(-((grid.RR - 0.735 * cfg.RSUN) / (0.02 * cfg.RSUN)) ** 2)
        Bph = np.ascontiguousarray(2.0e4 * prof * np.sin(2 * grid.TH))
        _, A_default = time_marching(Bph, np.zeros_like(Bph), 1.0e3, cfg, grid, setup)
        cfg.alpha_quenching = True
        _, A_on = time_marching(Bph, np.zeros_like(Bph), 1.0e3, cfg, grid, setup)
        assert np.array_equal(A_default, A_on)
