"""ana/ スクリプト共通のデータ読み込み処理。

各スクリプトは
    from ana_common import load_run
    run = load_run(datadir)
のように使う。datadir はコマンドライン第1引数でも指定できる
(get_datadir() を参照)。
"""
import os
import sys
from types import SimpleNamespace

import numpy as np

import S2MFD


def get_datadir(default='../data/'):
    """コマンドライン第1引数があればそれを datadir として返す。"""
    if len(sys.argv) > 1:
        return sys.argv[1]
    return default


def find_last_step(datadir):
    """datadir 内の最大のデータ番号を返す。"""
    n1 = 0
    for file in os.listdir(datadir):
        filel = file.split('.')
        if filel[0] == 'data':
            n1 = max(n1, int(filel[1]))
    return n1


def load_run(datadir, n0=0, n1=None, with_poloidal=True):
    """ラン一式 (設定・格子・全スナップショット) を読み込む。

    Parameters
    ----------
    datadir : str
        データディレクトリ
    n0, n1 : int
        読み込むデータ番号の範囲 [n0, n1)。n1=None なら最後まで。
    with_poloidal : bool
        True なら Brr, Bth も計算する。

    Returns
    -------
    SimpleNamespace
        data, cfg, grid, setup, timet, Bpht, Apht, (Brrt, Btht), tau_diff, n0, n1

    Notes
    -----
    メモリ使用量は (ixg, jxg, n1-n0) の float64 配列 2〜4 本分。
    128x128 で 5000 ステップなら 1 本あたり約 0.7 GB になるので注意。
    """
    data = S2MFD.Data.initial_load(datadir)
    cfg = data.cfg
    grid = data.grid
    setup = data.setup

    if n1 is None:
        n1 = find_last_step(datadir)

    tau_diff = cfg.RSUN**2 / cfg.ett
    nt = n1 - n0
    timet = np.zeros(nt)
    Bpht = np.zeros((grid.ixg, grid.jxg, nt))
    Apht = np.zeros((grid.ixg, grid.jxg, nt))
    Brrt = np.zeros((grid.ixg, grid.jxg, nt)) if with_poloidal else None
    Btht = np.zeros((grid.ixg, grid.jxg, nt)) if with_poloidal else None

    for n in range(n0, n1):
        data.data_load(n)
        k = n - n0
        timet[k] = data.time
        Bpht[:, :, k] = data.Bph
        Apht[:, :, k] = data.Aph
        if with_poloidal:
            Brr, Bth = S2MFD.physics.poloidal_mag(
                data.Aph, grid.RR, grid.sinTH, grid.drr, grid.dth)
            Brrt[:, :, k] = Brr
            Btht[:, :, k] = Bth

    return SimpleNamespace(
        data=data, cfg=cfg, grid=grid, setup=setup,
        timet=timet, Bpht=Bpht, Apht=Apht, Brrt=Brrt, Btht=Btht,
        tau_diff=tau_diff, n0=n0, n1=n1,
    )
