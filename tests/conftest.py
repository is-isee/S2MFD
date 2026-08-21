import os

os.environ.setdefault('MPLBACKEND', 'Agg')  # ヘッドレス環境でも main_loop の matplotlib import を通す

import numpy as np
import pytest

import S2MFD

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'golden')


def make_cfg(parameter_file='parameters/defaults.py', datadir=None, **overrides):
    """Cfg を生成し、datadir と任意の属性を上書きする補助関数。

    注意: defaults.py の派生量 (uu0, so0, ome など) は読込時に確定するため、
    ここで基本量 (rey, cso, ett など) を上書きしても派生量には反映されない。
    派生量を変えたい場合は派生量そのものを渡すこと。
    """
    cfg = S2MFD.Cfg(parameter_file)
    if datadir is not None:
        # 現実装はパスを文字列連結するので末尾スラッシュが必須
        cfg.datadir = str(datadir).rstrip('/') + '/'
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def make_grid(cfg):
    return S2MFD.Grid(
        ix=cfg.ix, jx=cfg.jx, margin=cfg.margin,
        rrmin=cfg.rrmin, rrmax=cfg.rrmax,
        thmin=cfg.thmin, thmax=cfg.thmax,
        stretch=getattr(cfg, 'grid_stretch', 0.0),
        stretch_center=getattr(cfg, 'grid_stretch_center',
                               0.5*(cfg.rrmin + cfg.rrmax)),
    )


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
