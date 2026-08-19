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

    def test_derived_parameters_frozen_after_load(self):
        """既知の仕様(罠): 基本量を後から変えても派生量は再計算されない。

        Phase 2 で cfg.resolve() を導入して明示的な再計算手段を提供する。
        このテストは現状の「変わらない」挙動を固定する(resolve() 導入後も
        resolve() を呼ばない限り変わらないことは同じ)。
        """
        cfg = S2MFD.Cfg()
        uu0_before = cfg.uu0
        cfg.rey = 1400
        assert cfg.uu0 == uu0_before  # 反映されない


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

    @pytest.mark.xfail(
        reason='既知バグ: Cfg.save() の型フィルタが numpy 型 (np.float32 等) を'
               '無警告で落とす (Phase 2 で修正予定)',
        strict=True)
    def test_save_preserves_numpy_scalars(self, tmp_path):
        cfg = make_cfg(datadir=tmp_path / 'data')
        os.makedirs(cfg.datadir, exist_ok=True)
        cfg.custom_gene = np.float32(1.5)
        cfg.save()
        loaded = S2MFD.Cfg.load(cfg.datadir + cfg.configfile)
        assert np.isclose(loaded.custom_gene, 1.5)


class TestCfgPath:
    @pytest.mark.xfail(
        reason='既知の制限: パラメタファイルのパスが常にパッケージディレクトリ'
               '相対に解決され、絶対パス・パッケージ外パスを渡せない'
               ' (Phase 2 で修正予定)',
        strict=True)
    def test_accepts_absolute_path(self, tmp_path):
        pfile = tmp_path / 'my_params.py'
        pfile.write_text('from S2MFD.parameters.defaults import *\nix = 42\n')
        cfg = S2MFD.Cfg(str(pfile))
        assert cfg.ix == 42
