"""Cfg クラスのテスト(パラメタ読込・save/load・既知の制限)。"""
import json
import os

import numpy as np
import pytest

import S2MFD
from conftest import make_cfg


class TestCfgLoad:
    def test_defaults(self):
        cfg = S2MFD.Cfg()
        assert cfg.ix == 128
        assert cfg.jx == 128
        assert cfg.margin == 1
        assert cfg.RSUN == 6.96e10
        assert cfg.alpha_type == 'BL'
        assert cfg.cont_flag is True

    def test_derived_parameters(self):
        cfg = S2MFD.Cfg()
        assert np.isclose(cfg.uu0, 700 * 1.e11 / 6.96e10)
        assert np.isclose(cfg.so0, 35 * 1.e11 / 6.96e10)
        assert np.isclose(cfg.omc, 0.92 * cfg.ome)

    def test_parameter_file_overrides(self):
        cfg = S2MFD.Cfg('parameters/alpha_omega.py')
        assert cfg.alpha_type == 'normal'
        assert cfg.uu0 == 0
        assert np.isclose(cfg.so0, 3.5 * 1.e11 / 6.96e10)
        # defaults から引き継がれた値
        assert cfg.ix == 128
        assert cfg.diffusive_type == 'J08'

    def test_derived_parameters_frozen_until_resolve(self):
        """基本量を後から変えても resolve() を呼ぶまで派生量は変わらない。"""
        cfg = S2MFD.Cfg()
        uu0_before = cfg.uu0
        cfg.rey = 1400
        assert cfg.uu0 == uu0_before  # resolve() を呼ぶまで反映されない


class TestCfgResolve:
    def test_resolve_updates_derived_from_primitives(self):
        cfg = S2MFD.Cfg()
        cfg.rey = 1400
        cfg.resolve()
        assert np.isclose(cfg.uu0, 1400 * cfg.ett / cfg.RSUN)
        cfg.cso = 70
        cfg.resolve()
        assert np.isclose(cfg.so0, 70 * cfg.ett / cfg.RSUN)

    def test_resolve_propagates_chain(self):
        """ett の変更が ome → omc, c2 まで伝播する。"""
        cfg = S2MFD.Cfg()
        cfg.ett = 2.e11
        cfg.resolve()
        assert np.isclose(cfg.ome, cfg.com / cfg.RSUN**2 * 2.e11)
        assert np.isclose(cfg.omc, 0.92 * cfg.ome)
        assert np.isclose(cfg.uu0, cfg.rey * 2.e11 / cfg.RSUN)

    def test_explicit_parameter_file_values_are_pinned(self):
        """hotta10.py のように導出式と異なる値を明示したものは再計算しない。"""
        cfg = S2MFD.Cfg('parameters/hotta10.py')
        ome_before = cfg.ome
        uu0_before = cfg.uu0
        cfg.resolve()
        assert cfg.ome == ome_before
        assert cfg.uu0 == uu0_before

    def test_user_assigned_derived_value_is_pinned(self):
        """GA 流儀で uu0 を直接代入した場合、resolve() が上書きしない。"""
        cfg = S2MFD.Cfg()
        cfg.uu0 = 500.0
        cfg.resolve()
        assert cfg.uu0 == 500.0
        # 以後も固定される
        cfg.rey = 1400
        cfg.resolve()
        assert cfg.uu0 == 500.0

    def test_hotta10_m_change_updates_c1d(self):
        """docstring の使用例 cfg.m = 2 の罠が resolve() で解消される。"""
        cfg = S2MFD.Cfg('parameters/hotta10.py')
        cfg.m = 2.0
        cfg.resolve()
        m, p, xi0 = 2.0, cfg.p, cfg.xi0
        assert np.isclose(cfg.c1d, (2*m+1)*(m+p)/(m+1)/p * xi0**(-m))


class TestCfgSaveLoad:
    def test_roundtrip(self, tmp_path):
        cfg = make_cfg(datadir=tmp_path / 'data')
        os.makedirs(cfg.datadir, exist_ok=True)
        cfg.save()
        loaded = S2MFD.Cfg.load(cfg.datadir + cfg.configfile)
        assert loaded.ix == cfg.ix
        assert loaded.alpha_type == cfg.alpha_type
        assert loaded.uu0 == cfg.uu0

    def test_save_is_valid_json(self, tmp_path):
        cfg = make_cfg(datadir=tmp_path / 'data')
        os.makedirs(cfg.datadir, exist_ok=True)
        cfg.save()
        with open(cfg.datadir + cfg.configfile) as f:
            params = json.load(f)
        assert params['ix'] == 128

    def test_save_preserves_numpy_scalars(self, tmp_path):
        # Phase 2 で修正済み: 保存前に numpy スカラーをネイティブ型へ変換
        cfg = make_cfg(datadir=tmp_path / 'data')
        os.makedirs(cfg.datadir, exist_ok=True)
        cfg.custom_gene = np.float32(1.5)
        cfg.save()
        loaded = S2MFD.Cfg.load(cfg.datadir + cfg.configfile)
        assert np.isclose(loaded.custom_gene, 1.5)


class TestCfgPath:
    def test_accepts_absolute_path(self, tmp_path):
        # Phase 2 で修正済み: 存在するパスはそのまま使う
        pfile = tmp_path / 'my_params.py'
        pfile.write_text('from S2MFD.parameters.defaults import *\nix = 42\n')
        cfg = S2MFD.Cfg(str(pfile))
        assert cfg.ix == 42
