"""Simulation の統合テスト(小規模ラン・characterization)。

ゴールデンデータは tests/golden/generate_golden.py で生成する。
設定は conftest.small_cfg と同一 (alpha_omega, 32x32, tend=200d, dtout=40d)。
"""
import glob
import os

import numpy as np
import pytest

import S2MFD
from conftest import make_cfg, run_short_simulation


class TestCflCondition:
    def test_matches_vectorized_formula(self, small_cfg):
        sim = S2MFD.Simulation(small_cfg)
        sim.cfl_condition()
        grid, setup = sim.grid, sim.setup
        m = grid.margin
        rr = grid.rr[m:grid.ixg - m, None]
        cell = np.minimum(grid.drr[m:grid.ixg - m, None], rr * grid.dth)
        uu = np.sqrt(setup.urr[m:-m, m:-m]**2 + setup.uth[m:-m, m:-m]**2)
        dt_adv = 0.8 * cell / np.sqrt(uu**2 + 1e-20)
        dt_dif = 0.8 * cell**2 / (2 * setup.et[m:-m, m:-m] + 1e-20)
        expected = min(1.e10, dt_adv.min(), dt_dif.min())
        assert np.isclose(sim.dt, expected)

    def test_dt_positive_and_reasonable(self, small_cfg):
        sim = S2MFD.Simulation(small_cfg)
        sim.cfl_condition()
        # 32x32 では拡散律速: dt ~ 0.8*drr^2/(2*ett)
        drr = (small_cfg.rrmax - small_cfg.rrmin) / small_cfg.ix
        assert 0 < sim.dt <= 0.8 * drr**2 / (2 * small_cfg.ett) * 1.0001


class TestShortRun:
    def test_run_completes_and_writes_snapshots(self, small_cfg):
        sim = run_short_simulation(small_cfg)
        files = sorted(glob.glob(small_cfg.datadir + 'data.*.npz'))
        assert len(files) == sim.nd + 1  # nd=0 (初期) + 各出力
        assert sim.nd >= 4  # tend=200d, dtout=40d
        assert np.all(np.isfinite(sim.Bph))
        assert np.all(np.isfinite(sim.Aph))
        # 磁場が実際に生成されている (α効果で Bph が種から成長)
        assert np.max(np.abs(sim.Bph)) > 0

    def test_metadata_files_written(self, small_cfg):
        run_short_simulation(small_cfg)
        for f in (small_cfg.gridfile, small_cfg.setupfile,
                  small_cfg.legendrefile, small_cfg.configfile):
            assert os.path.exists(small_cfg.datadir + f)

    def test_matches_golden(self, small_cfg, golden_dir):
        path = os.path.join(golden_dir, 'short_run.npz')
        if not os.path.exists(path):
            pytest.skip(f'golden データ未生成: {path} (generate_golden.py を実行)')
        golden = np.load(path)
        sim = run_short_simulation(small_cfg)
        assert sim.n == int(golden['n'])
        assert sim.nd == int(golden['nd'])
        assert np.isclose(float(sim.time), float(golden['time']), rtol=1e-12)
        assert np.allclose(sim.Bph, golden['Bph'], rtol=1e-10, atol=1e-300)
        assert np.allclose(sim.Aph, golden['Aph'], rtol=1e-10, atol=1e-300)


class TestSnapshotTimestamp:
    def test_snapshot_label_matches_state(self, small_cfg, tmp_path):
        # Phase 2 で修正済み: 「積分 → 時刻更新 → 出力」の順に変更
        sim = run_short_simulation(small_cfg)
        snap = np.load(sim.get_data_file_path(1))

        # 同一設定で手動積分し「ラベル時刻まで積分した場」を作る
        cfg2 = make_cfg(
            'parameters/alpha_omega.py',
            datadir=tmp_path / 'data_manual',
            ix=small_cfg.ix, jx=small_cfg.jx,
            tend=small_cfg.tend, dtout=small_cfg.dtout,
        )
        sim2 = S2MFD.Simulation(cfg2)
        sim2.initialize_simulation()
        sim2.cfl_condition()
        sim2.initial_condition()
        assert np.isclose(sim2.dt, sim.dt)

        time, dtout = 0.0, cfg2.dtout
        while True:
            time += sim2.dt
            sim2.tvd_runge_kutta()  # 正しい実装では出力前にここまで積分される
            if time // dtout != (time - sim2.dt) // dtout:
                break
        assert np.isclose(float(snap['time']), time)
        assert np.allclose(snap['Bph'], sim2.Bph)
        assert np.allclose(snap['Aph'], sim2.Aph)


class TestRestart:
    def test_restart_continues_run(self, small_cfg):
        # 2026-08-19 確認: 0次元配列問題 (test_data_load_returns_python_scalars)
        # があっても現行の numpy/numba ではリスタートは動作する。
        # 型の健全化は Phase 2 で行うが、このテストは常時パスすべき回帰テスト。
        run_short_simulation(small_cfg)

        cfg2 = make_cfg(
            'parameters/alpha_omega.py',
            datadir=small_cfg.datadir,
            ix=small_cfg.ix, jx=small_cfg.jx,
            tend=small_cfg.tend + 100 * 86400,
            dtout=small_cfg.dtout,
        )
        assert cfg2.cont_flag is True
        sim2 = S2MFD.Simulation(cfg2)
        sim2.initialize_simulation()  # 既存 datadir → npz から grid/setup を復元
        sim2.cfl_condition()
        sim2.initial_condition()      # 最新スナップショットから再開
        sim2.main_loop()
        assert float(sim2.time) >= cfg2.tend
        assert np.all(np.isfinite(sim2.Bph))

    def test_data_load_returns_python_scalars(self, small_cfg):
        # Phase 2 で修正済み: data_load が float()/int() で復元する
        sim = run_short_simulation(small_cfg)
        sim.data_load(0)
        assert not isinstance(sim.time, np.ndarray)
        assert not isinstance(sim.n, np.ndarray)


class TestResumeConsistency:
    def test_mismatched_config_raises(self, small_cfg, tmp_path):
        """物理パラメタを変えて既存 datadir を再利用すると明示エラーになる。"""
        run_short_simulation(small_cfg)
        cfg2 = make_cfg(
            'parameters/alpha_omega.py',
            datadir=small_cfg.datadir,
            ix=small_cfg.ix, jx=small_cfg.jx,
            tend=small_cfg.tend, dtout=small_cfg.dtout,
            cso=99.0,  # 物理パラメタを変更
        )
        assert cfg2.cont_flag is True
        sim2 = S2MFD.Simulation(cfg2)
        with pytest.raises(RuntimeError, match='cso'):
            sim2.initialize_simulation()

    def test_tend_change_is_allowed(self, small_cfg):
        """tend/dtout の変更 (ランの延長) は再開として許容される。"""
        run_short_simulation(small_cfg)
        cfg2 = make_cfg(
            'parameters/alpha_omega.py',
            datadir=small_cfg.datadir,
            ix=small_cfg.ix, jx=small_cfg.jx,
            tend=small_cfg.tend + 40 * 86400,
            dtout=small_cfg.dtout,
        )
        sim2 = S2MFD.Simulation(cfg2)
        sim2.initialize_simulation()  # エラーにならない


class TestCheckFinite:
    def test_nan_raises(self, small_cfg):
        sim = S2MFD.Simulation(small_cfg)
        sim.initialize_simulation()
        sim.cfl_condition()
        sim.initial_condition()
        sim.Bph[5, 5] = np.nan
        with pytest.raises(RuntimeError, match='[Nn]on-finite'):
            sim.check_finite()


class TestRunSimulationApi:
    def test_run_simulation_with_cfg(self, tmp_path):
        cfg = make_cfg(
            'parameters/alpha_omega.py',
            datadir=tmp_path / 'data',
            ix=16, jx=16,
            tend=100 * 86400,
            dtout=40 * 86400,
        )
        result = S2MFD.run_simulation(cfg)
        assert os.path.exists(cfg.datadir + 'config.json')
        # Phase 2 で修正済み: Simulation オブジェクトを返す
        assert isinstance(result, S2MFD.Simulation)
        assert np.all(np.isfinite(result.Bph))
