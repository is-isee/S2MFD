import os

os.environ.setdefault('MPLBACKEND', 'Agg')  # ヘッドレス環境でも main_loop の matplotlib import を通す

import numpy as np
import pytest

import S2MFD

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'golden')


def make_cfg(parameter_file='parameters/defaults.py', datadir=None,
             **overrides):
    """:func:`S2MFD.build_cfg` の別名 (既存テストとの互換のために残す)。

    **新しいコードは ``S2MFD.build_cfg`` を直接使うこと。**
    実行スクリプトが ``tests/`` を import しなければならないのはおかしいので、
    2026-08-23 に本体へ移した。
    """
    return S2MFD.build_cfg(parameter_file, datadir=datadir, **overrides)


def make_grid(cfg):
    """:meth:`S2MFD.Grid.from_cfg` の別名 (既存テストとの互換)。"""
    return S2MFD.Grid.from_cfg(cfg)


def run_short_simulation(cfg):
    """main_loop までを一通り実行して Simulation を返す。"""
    sim = S2MFD.Simulation(cfg)
    sim.initialize_simulation()
    sim.cfl_condition()
    sim.initial_condition()
    sim.main_loop()
    return sim


@pytest.fixture
def small_cfg(tmp_path):
    """32x32 の小規模設定 (alpha_omega ベース、子午面流なし)。"""
    return make_cfg(
        'parameters/alpha_omega.py',
        datadir=tmp_path / 'data',
        ix=32, jx=32,
        tend=200 * 86400,
        dtout=40 * 86400,
    )


@pytest.fixture
def golden_dir():
    return GOLDEN_DIR
