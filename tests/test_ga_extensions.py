"""パラメタ推定 (GA) 用に追加した Simulation/Setup/tools API のテスト。"""
import glob

import numpy as np
import pytest

import S2MFD
from S2MFD.tools import sunspot_proxy
from conftest import make_cfg, make_grid, run_short_simulation


class TestSunspotProxy:
    def test_value(self):
        cfg = make_cfg(ix=32, jx=32)
        grid = make_grid(cfg)
        Bph = np.zeros((grid.ixg, grid.jxg))
        base = 1 + np.argmin(abs(grid.rr - 0.7 * cfg.RSUN))
        loca = np.argmin(abs(grid.th - 75 / 180 * np.pi))
        Bph[base, loca] = 2.0
        assert np.isclose(sunspot_proxy(Bph, grid, cfg), 5.8653520852 * 4.0)

    def test_custom_gamma(self):
        cfg = make_cfg(ix=16, jx=16)
        grid = make_grid(cfg)
        Bph = np.ones((grid.ixg, grid.jxg))
        assert np.isclose(sunspot_proxy(Bph, grid, cfg, gamma=2.0), 2.0)


class TestSetupPartialRebuild:
    def test_build_flow_scales_with_uu0(self):
        cfg = make_cfg()  # defaults: J08 flow, uu0 > 0
        grid = make_grid(cfg)
        setup = S2MFD.Setup(cfg, grid)
        urr0 = setup.urr.copy()
        cfg.uu0 = cfg.uu0 * 2
        setup.build_flow(cfg, grid)
        assert np.allclose(setup.urr, 2 * urr0)

    def test_build_alpha_scales_with_so0(self):
        cfg = make_cfg()  # defaults: BL alpha
        grid = make_grid(cfg)
        setup = S2MFD.Setup(cfg, grid)
        so_before = setup.so.copy()
        om_before = setup.om.copy()
        cfg.so0 = cfg.so0 * 3
        setup.build_alpha(cfg, grid)
        assert np.allclose(setup.so, 3 * so_before)
        # 他のプロファイルは不変
        assert np.array_equal(setup.om, om_before)


class TestTimeDependentHooks:
    def _sim(self, tmp_path, **overrides):
        cfg = make_cfg(
            'parameters/defaults.py',
            datadir=tmp_path / 'data',
            ix=32, jx=32,
            tend=200 * 86400, dtout=40 * 86400,
            **overrides,
        )
        sim = S2MFD.Simulation(cfg)
        sim.initialize_simulation()
        sim.cfl_condition()
        sim.initial_condition()
        return sim

    def test_hooks_update_amplitudes_and_setup(self, tmp_path):
        sim = self._sim(tmp_path)
        sim.cfg.uu0_of_time = lambda t: 500.0 + t / 86400
        sim.cfg.so0_of_time = lambda t: 40.0
        sim.time = 86400.0
        sim.update_time_dependent_parameters()
        assert np.isclose(sim.cfg.uu0, 501.0)
        assert np.isclose(sim.cfg.so0, 40.0)
        # setup も更新されている (J08 流れは uu0 に線形)
        cfg2 = make_cfg('parameters/defaults.py', ix=32, jx=32, uu0=501.0)
        setup2 = S2MFD.Setup(cfg2, sim.grid)
        assert np.allclose(sim.setup.urr, setup2.urr)

    def test_hook_updates_cfl(self, tmp_path):
        sim = self._sim(tmp_path)
        dt_before = sim.dt
        sim.cfg.uu0_of_time = lambda t: 100000.0  # 非現実的な高速流
        sim.update_time_dependent_parameters()
        assert sim.dt < dt_before  # 移流律速で dt が縮む


class TestRunWindow:
    def test_records_series_via_callback(self, tmp_path):
        cfg = make_cfg(
            'parameters/alpha_omega.py',
            datadir=tmp_path / 'data',
            ix=32, jx=32,
            tend=200 * 86400, dtout=40 * 86400,
        )
        sim = S2MFD.Simulation(cfg)
        sim.initialize_simulation()
        sim.cfl_condition()
        sim.initial_condition()

        sn, ts = [], []

        def record(s):
            sn.append(sunspot_proxy(s.Bph, s.grid, s.cfg))
            ts.append(s.time)

        sim.run_window(200 * 86400, on_output=record, save_output=False)
        assert len(sn) == sim.nd  # 初期スナップショット (nd=0) を除く出力数
        assert all(np.isfinite(sn))
        assert ts == sorted(ts)
        # save_output=False: nd=0 以外のスナップショットは書かれない
        files = glob.glob(cfg.datadir + 'data.*.npz')
        assert len(files) == 1

    def test_matches_main_loop(self, small_cfg, tmp_path):
        """run_window(tend) は main_loop と同じ結果を与える。"""
        sim1 = run_short_simulation(small_cfg)

        cfg2 = make_cfg(
            'parameters/alpha_omega.py',
            datadir=tmp_path / 'data2',
            ix=small_cfg.ix, jx=small_cfg.jx,
            tend=small_cfg.tend, dtout=small_cfg.dtout,
        )
        sim2 = S2MFD.Simulation(cfg2)
        sim2.initialize_simulation()
        sim2.cfl_condition()
        sim2.initial_condition()
        sim2.run_window(cfg2.tend)
        assert np.array_equal(sim1.Bph, sim2.Bph)
        assert np.array_equal(sim1.Aph, sim2.Aph)
        assert sim1.n == sim2.n


class TestSaveExtraFields:
    def test_snapshot_contains_amplitudes(self, small_cfg):
        sim = run_short_simulation(small_cfg)
        d = np.load(sim.get_data_file_path(sim.nd))
        for key in ('nd', 'dt', 'uu0', 'so0'):
            assert key in d.files
        assert int(d['nd']) == sim.nd
        assert np.isclose(float(d['uu0']), float(small_cfg.uu0))


class TestSpinUp:
    def test_spin_up_finds_minimum(self, tmp_path):
        cfg = make_cfg(
            'parameters/alpha_omega.py',
            datadir=tmp_path / 'data',
            ix=32, jx=32,
            tend=200 * 86400, dtout=40 * 86400,
        )
        sim = S2MFD.Simulation(cfg)
        sim.initialize_simulation()
        sim.cfl_condition()
        sim.initial_condition()
        duration = 2 * 365 * 86400
        sim.spin_up(duration, until_minimum=True)
        assert sim.time >= duration
        assert np.all(np.isfinite(sim.Bph))

    def test_set_field(self, tmp_path):
        cfg = make_cfg('parameters/alpha_omega.py',
                       datadir=tmp_path / 'data', ix=16, jx=16)
        sim = S2MFD.Simulation(cfg)
        Bph = np.random.default_rng(0).standard_normal((sim.grid.ixg, sim.grid.jxg))
        Aph = np.zeros_like(Bph)
        sim.set_field(Bph, Aph, time=1.5e9, n=100, nd=7)
        assert np.array_equal(sim.Bph, Bph)
        assert sim.time == 1.5e9
        assert sim.n == 100 and sim.nd == 7
        # コピーであること (元配列の変更が波及しない)
        Bph[0, 0] = 999
        assert sim.Bph[0, 0] != 999
