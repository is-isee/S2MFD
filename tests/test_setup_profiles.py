"""Setup(背景場プロファイル)のテスト。

ゴールデンデータ (tests/golden/setup_*.npz) は tests/golden/generate_golden.py で
生成する。物理を変更しない限りこれらの配列は一致し続けるはず。
"""
import os

import numpy as np
import pytest

import S2MFD
from conftest import make_cfg, make_grid


def _setup_for(parameter_file, ix=32, jx=32):
    cfg = make_cfg(parameter_file, ix=ix, jx=jx)
    grid = make_grid(cfg)
    return cfg, grid, S2MFD.Setup(cfg, grid)


class TestSetupProperties:
    def test_ibase_is_tachocline_index(self):
        cfg, grid, setup = _setup_for('parameters/defaults.py')
        assert setup.ibase == np.argmin(abs(grid.rr - cfg.rrc))

    def test_diffusivity_limits_j08(self):
        cfg, grid, setup = _setup_for('parameters/defaults.py')
        # 深部で etc、表面近くで ett に漸近
        # (下端 0.65R は erf 遷移 (rrc=0.7R, d=0.02R) の裾なので ~8% 残差がある)
        assert np.isclose(setup.et[grid.margin, 0], cfg.etc, rtol=0.1)
        assert np.isclose(setup.et[-1, 0], cfg.ett, rtol=1e-2)

    def test_omega_limits_j08(self):
        cfg, grid, setup = _setup_for('parameters/defaults.py')
        # 放射層は剛体回転 omc
        assert np.allclose(setup.om[grid.margin, :], cfg.omc, rtol=1e-3)
        # 表面の赤道 (θ=π/2 付近) は ome に近い
        jeq = grid.jxg // 2
        assert np.isclose(setup.om[-1, jeq], cfg.ome, rtol=1e-2)

    def test_meridional_flow_zero_when_uu0_zero(self):
        cfg, grid, setup = _setup_for('parameters/alpha_omega.py')
        assert np.all(setup.urr == 0)
        assert np.all(setup.uth == 0)

    def test_meridional_flow_boundary_antisymmetry(self):
        cfg, grid, setup = _setup_for('parameters/defaults.py')
        m = grid.margin
        # 動径境界: urr が反対称
        assert np.allclose(setup.urr[m - 1, :], -setup.urr[m, :])
        assert np.allclose(setup.urr[grid.ixg - m, :], -setup.urr[grid.ixg - m - 1, :])
        # 緯度境界: uth が反対称
        assert np.allclose(setup.uth[:, m - 1], -setup.uth[:, m])
        assert np.allclose(setup.uth[:, grid.jxg - m], -setup.uth[:, grid.jxg - m - 1])

    def test_alpha_profile_antisymmetric_about_equator(self):
        """BL/normal 型 α は cosθ 因子により赤道反対称。"""
        cfg, grid, setup = _setup_for('parameters/defaults.py')
        m = grid.margin
        so_in = setup.so[m:grid.ixg - m, m:grid.jxg - m]
        assert np.allclose(so_in, -so_in[:, ::-1], atol=1e-12 * np.max(np.abs(so_in)))


class TestSetupGolden:
    @pytest.mark.parametrize('name', ['defaults', 'alpha_omega', 'hotta10'])
    def test_profiles_match_golden(self, golden_dir, name):
        path = os.path.join(golden_dir, f'setup_{name}.npz')
        if not os.path.exists(path):
            pytest.skip(f'golden データ未生成: {path} (generate_golden.py を実行)')
        golden = np.load(path)
        cfg, grid, setup = _setup_for(f'parameters/{name}.py')
        for key in ('om', 'omrr', 'omth', 'et', 'etrr', 'so', 'urr', 'uth'):
            assert np.allclose(getattr(setup, key), golden[key], rtol=1e-12, atol=0), \
                f'{name}:{key} がゴールデンデータと不一致'
        assert int(golden['ibase']) == setup.ibase


class TestSetupSaveLoad:
    def test_arrays_roundtrip(self, tmp_path):
        cfg, grid, setup = _setup_for('parameters/defaults.py')
        path = str(tmp_path / 'setup.npz')
        setup.save(path)
        loaded = S2MFD.Setup.load(path)
        assert np.allclose(loaded.om, setup.om)
        assert np.allclose(loaded.so, setup.so)

    def test_ibase_keeps_python_type(self, tmp_path):
        # Phase 2 で修正済み: NpzIO.load が0次元配列を .item() で復元する
        cfg, grid, setup = _setup_for('parameters/defaults.py')
        path = str(tmp_path / 'setup.npz')
        setup.save(path)
        loaded = S2MFD.Setup.load(path)
        assert not isinstance(loaded.ibase, np.ndarray)
