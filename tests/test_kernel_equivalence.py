"""高速化カーネルが参照実装と等価であることを検証する。

方針:
- ``cfg.exact_arithmetic = True`` の経路は参照実装と**ビット一致**すること
- 既定の高速経路は 1 substep あたり倍精度 1 ULP 程度 (相対 ~1e-15) の差に
  収まり、全積分してもそれが増幅しないこと
"""
import numpy as np
import pytest

import S2MFD
from S2MFD.physics import time_marching, time_marching_reference
from S2MFD.physics.physics_core import (separable_profiles, _rank1,
                                        get_time_marching_kernel)
from conftest import make_cfg, make_grid

PARAM_FILES = [
    ('parameters/inference.py', 'BL'),
    ('parameters/alpha_omega.py', 'normal'),
    ('parameters/hotta10.py', 'H10'),
    ('parameters/sample.py', 'BL'),
]


def _case(parameter_file, ix=48, jx=48, seed=1):
    cfg = make_cfg(parameter_file, ix=ix, jx=jx)
    grid = make_grid(cfg)
    setup = S2MFD.Setup(cfg, grid)
    rng = np.random.default_rng(seed)
    Bph = rng.standard_normal((grid.ixg, grid.jxg)) * 0.5
    Aph = rng.standard_normal((grid.ixg, grid.jxg)) * 1e7
    return cfg, grid, setup, Bph, Aph


class TestExactPathIsBitIdentical:
    @pytest.mark.parametrize('parameter_file,alpha', PARAM_FILES)
    def test_matches_reference(self, parameter_file, alpha):
        cfg, grid, setup, Bph, Aph = _case(parameter_file)
        assert cfg.alpha_type == alpha
        Br, Ar = time_marching_reference(Bph, Aph, 1e5, cfg, grid, setup)
        cfg.exact_arithmetic = True
        Be, Ae = time_marching(Bph, Aph, 1e5, cfg, grid, setup)
        assert np.array_equal(Br, Be), 'Bph がビット一致しない'
        assert np.array_equal(Ar, Ae), 'Aph がビット一致しない'


class TestFastPathAccuracy:
    @pytest.mark.parametrize('parameter_file,alpha', PARAM_FILES)
    def test_within_one_ulp(self, parameter_file, alpha):
        cfg, grid, setup, Bph, Aph = _case(parameter_file)
        Br, Ar = time_marching_reference(Bph, Aph, 1e5, cfg, grid, setup)
        Bf, Af = time_marching(Bph, Aph, 1e5, cfg, grid, setup)
        rel_b = np.max(np.abs(Br - Bf)) / np.max(np.abs(Br))
        rel_a = np.max(np.abs(Ar - Af)) / np.max(np.abs(Ar))
        assert rel_b < 1e-14, f'Bph の相対差が大きすぎる: {rel_b:.2e}'
        assert rel_a < 1e-14, f'Aph の相対差が大きすぎる: {rel_a:.2e}'

    def test_alpha_source_is_not_dropped(self):
        """局所 alpha ('normal') でソース項が消えていないこと。

        分離可能性の最適化で Aph のソース項をゼロにしてしまう回帰があった。
        """
        cfg, grid, setup, Bph, Aph = _case('parameters/alpha_omega.py')
        Aph = np.zeros_like(Aph)          # ソース項だけが Aph を動かす状況
        _, Af = time_marching(Bph, Aph, 1e5, cfg, grid, setup)
        assert np.max(np.abs(Af)) > 0, 'alpha 効果のソース項が効いていない'


class TestSeparableDecomposition:
    def test_rank1_roundtrip(self):
        u = np.linspace(1.0, 2.0, 7)
        v = np.linspace(-1.0, 3.0, 5)
        dec = _rank1(np.outer(u, v))
        assert dec is not None
        assert np.allclose(np.outer(*dec), np.outer(u, v))

    def test_rank1_rejects_non_separable(self):
        arr = np.arange(35, dtype=float).reshape(7, 5)
        arr[3, 2] += 100.0        # rank を上げる
        assert _rank1(arr) is None

    def test_rank1_handles_zero_array(self):
        dec = _rank1(np.zeros((4, 3)))
        assert dec is not None
        assert np.allclose(np.outer(*dec), 0.0)

    @pytest.mark.parametrize('parameter_file,alpha', PARAM_FILES)
    def test_profiles_are_separable(self, parameter_file, alpha):
        """同梱の背景場プロファイルはすべて f(r)*g(theta) の形になっている。"""
        cfg, grid, setup, _, _ = _case(parameter_file)
        sep, _ = separable_profiles(setup)
        assert sep, f'{parameter_file} が分離可能と判定されなかった'

    def test_separable_and_general_kernels_agree(self):
        """分離版と一般版のカーネルが同じ結果を与えること。"""
        cfg, grid, setup, Bph, Aph = _case('parameters/inference.py')
        Bs, As = time_marching(Bph, Aph, 1e5, cfg, grid, setup)
        # 分離不可能にして一般版を強制する
        setup.omth = setup.omth + 1e-30*np.arange(setup.omth.size).reshape(
            setup.omth.shape)
        sep, _ = separable_profiles(setup)
        assert not sep
        Bg, Ag = time_marching(Bph, Aph, 1e5, cfg, grid, setup)
        assert np.allclose(Bs, Bg, rtol=1e-13)
        assert np.allclose(As, Ag, rtol=1e-13)


class TestSteppingLoop:
    def test_step_count_and_fields_match_manual(self, tmp_path):
        """numba ループが 1 ステップずつ回した場合と一致すること。"""
        cfg = make_cfg('parameters/alpha_omega.py', datadir=tmp_path / 'a',
                       ix=32, jx=32, tend=200 * 86400, dtout=40 * 86400)
        sim = S2MFD.Simulation(cfg)
        sim.initialize_simulation()
        sim.cfl_condition()
        sim.initial_condition()

        cfg2 = make_cfg('parameters/alpha_omega.py', datadir=tmp_path / 'b',
                        ix=32, jx=32, tend=200 * 86400, dtout=40 * 86400)
        sim2 = S2MFD.Simulation(cfg2)
        sim2.initialize_simulation()
        sim2.cfl_condition()
        sim2.initial_condition()

        t_stop = sim.time + 30 * sim.dt
        n_batch = S2MFD.physics.stepping.advance_to(sim, t_stop)
        n_manual = 0
        while sim2.time < t_stop:
            sim2.tvd_runge_kutta()
            sim2.time += sim2.dt
            sim2.n += 1
            n_manual += 1

        assert n_batch == n_manual
        assert np.isclose(sim.time, sim2.time, rtol=1e-15)
        assert np.allclose(sim.Bph, sim2.Bph, rtol=1e-12)
        assert np.allclose(sim.Aph, sim2.Aph, rtol=1e-12)
