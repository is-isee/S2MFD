"""長時間ベンチマークテスト (既定では実行されない: pytest -m slow で実行)。

Jouve et al. (2008) ベンチマークの簡易版。厳密な成長率・周期の検証は
doc/dev_records の検証記録と ana/J08_test.py を参照。
"""
import numpy as np
import pytest

import S2MFD
from conftest import make_cfg, run_short_simulation

pytestmark = pytest.mark.slow


def test_alpha_omega_dynamo_grows(tmp_path):
    """αΩ設定 (超臨界) で磁場が指数成長すること。"""
    cfg = make_cfg(
        'parameters/alpha_omega.py',
        datadir=tmp_path / 'data',
        ix=64, jx=64,
        tend=4000 * 86400,
        dtout=400 * 86400,
    )
    sim = run_short_simulation(cfg)
    assert np.all(np.isfinite(sim.Bph))
    d_early = np.load(sim.get_data_file_path(1))
    d_late = np.load(sim.get_data_file_path(sim.nd))
    amp_early = np.max(np.abs(d_early['Bph']))
    amp_late = np.max(np.abs(d_late['Bph']))
    assert amp_late > amp_early


def test_potential_bc_run_is_stable(tmp_path):
    """potential 上部境界条件でのランが有限のまま進むこと。"""
    cfg = make_cfg(
        'parameters/upper_potential.py',
        datadir=tmp_path / 'data',
        ix=64, jx=64,
        tend=2000 * 86400,
        dtout=400 * 86400,
    )
    sim = run_short_simulation(cfg)
    assert np.all(np.isfinite(sim.Bph))
    assert np.all(np.isfinite(sim.Aph))
