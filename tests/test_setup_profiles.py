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


class TestForcingRadiusDecoupling:
    """Λ 効果と α 効果の r_max を領域上端から切り離す。

    Rempel (2005) 式 (33) の Λ 効果は
    :math:`f\\propto\\tanh((r_{\\max}-r)/d)`、Rempel (2006) 式 (17) の α 効果は
    :math:`f_\\alpha=\\max[0,1-(r-r_{\\max})^2/d_\\alpha^2]` で、どちらも
    論文の領域上端 :math:`r_{\\max}=0.985R_\\odot` を基準にしている。

    実装は長らくこれを ``grid.rrmax`` (計算領域の上端) に結びつけていたため、
    数値安定性のために領域を 0.96 R⊙ に切り詰めると駆動の位置まで
    0.025 R⊙ 内側に動いてしまっていた
    (α ソースの下限が論文の 0.935 R⊙ から 0.91 R⊙ になる)。
    """

    def _profiles(self, rrmax_frac, r_max=None):
        over = {'rrmax': rrmax_frac * 6.96e10, 'dynamics': 'dynamic'}
        if r_max is not None:
            over['r_max'] = r_max * 6.96e10
        cfg = make_cfg('parameters/rempel06.py', ix=64, jx=32, **over)
        grid = make_grid(cfg)
        return cfg, grid, S2MFD.Setup(cfg, grid)

    def test_default_follows_domain_top(self):
        """r_max を指定しなければ従来どおり領域上端に追随する。"""
        cfg, grid, a = self._profiles(0.96)
        _, _, b = self._profiles(0.96, r_max=0.96)
        assert np.array_equal(a.lam_rp, b.lam_rp)
        assert np.array_equal(a.so, b.so)

    def test_alpha_source_edge_follows_r_max_not_domain(self):
        """α ソースの内端は r_max - d_alpha に来る (領域上端ではない)。"""
        cfg, grid, s = self._profiles(0.96, r_max=0.985)
        m = grid.margin
        rr = grid.rr[m:grid.ixg - m]
        # so が立っている最も内側の半径
        active = np.abs(s.so[m:grid.ixg - m]).max(axis=1) > 0
        r_edge = rr[active][0] / cfg.RSUN
        assert abs(r_edge - (0.985 - 0.05)) < 0.01, \
            f"α ソースの内端が {r_edge:.3f} R (期待 0.935 R)"

    def test_domain_coupled_alpha_edge_is_wrong(self):
        """従来 (r_max=領域上端) だと内端が 0.91 R にずれることの記録。"""
        cfg, grid, s = self._profiles(0.96)
        m = grid.margin
        rr = grid.rr[m:grid.ixg - m]
        active = np.abs(s.so[m:grid.ixg - m]).max(axis=1) > 0
        r_edge = rr[active][0] / cfg.RSUN
        assert abs(r_edge - (0.96 - 0.05)) < 0.01

    def test_warns_when_r_max_differs_from_domain_top(self):
        """r_max != 領域上端では Λ フラックスが上部境界で消えないので警告する。

        Rempel (2005) §2.5 は "we require a vanishing angular momentum flux at
        the top boundary" と述べており、tanh((r_max-r)/d) はそのための遷移層。
        r_max を領域上端からずらすと、この条件が破れる。
        """
        with pytest.warns(UserWarning, match='r_max'):
            self._profiles(0.96, r_max=0.985)

    def test_no_warning_when_consistent(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            self._profiles(0.985, r_max=0.985)
