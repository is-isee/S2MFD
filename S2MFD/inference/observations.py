"""観測データ (黒点数時系列) の読み込み・内挿・極小期検出。

データ規約 (旧 IDPA の make_time_series.py と同一):

- 年単位の時系列を 40 日間隔に線形内挿して使う
  (シミュレーションの出力間隔 dtout = 40 日と 1:1 対応させるため)
- 時刻の単位は「年 × 365 日 × 86400 秒」
"""
import os
from dataclasses import dataclass
from importlib import resources

import numpy as np
from scipy.signal import argrelextrema

__all__ = ['Observations', 'interpolate_to_interval', 'load_observations',
           'detect_minimum_years', 'load_initial_field']

YEAR_SECONDS = 365 * 24 * 3600
DEFAULT_INTERVAL_DAYS = 40


def _package_data_path(filename):
    return resources.files('S2MFD.inference').joinpath('data', filename)


@dataclass
class Observations:
    """40日間隔に内挿済みの黒点数観測時系列。

    Attributes
    ----------
    years : numpy.ndarray
        年 (小数)
    seconds : numpy.ndarray
        絶対時刻 [s] (= years * 365 * 86400)。シミュレーション時刻と直接
        比較する。
    ssn : numpy.ndarray
        黒点数 (内挿済み)
    """
    years: np.ndarray
    seconds: np.ndarray
    ssn: np.ndarray

    def index_of_year(self, year):
        """指定した年に最も近いデータ点のインデックスを返す。"""
        return int(np.argmin(np.abs(self.years - year)))

    def detect_minima_indices(self, order=27):
        """黒点数の極小期のインデックスを返す。

        order=27 は 27 点 × 40 日 ≈ 2.96 年の窓で前後比較する
        (旧 detect_minimum.py と同じ)。平坦区間対策として微小な
        単調増加を加えてタイブレークする。
        """
        eps = 1e-6 * np.arange(len(self.ssn))
        return argrelextrema(self.ssn + eps, np.less, order=order)[0]

    def detect_minima_years(self, order=27):
        """黒点数の極小期の年のリストを返す。"""
        return self.years[self.detect_minima_indices(order=order)]


def interpolate_to_interval(years, ssn, interval_days=DEFAULT_INTERVAL_DAYS):
    """年単位の時系列を interval_days 間隔に線形内挿する。"""
    interval = interval_days / 365.0
    years_interp = np.arange(years[0], years[-1], interval)
    seconds_interp = years_interp * YEAR_SECONDS
    ssn_interp = np.interp(years_interp, years, ssn)
    return Observations(years=years_interp, seconds=seconds_interp, ssn=ssn_interp)


def load_observations(source='OBS', time_col=0, ssn_col=1,
                      interval_days=DEFAULT_INTERVAL_DAYS):
    """観測データを読み込み、40日間隔の Observations を返す。

    Parameters
    ----------
    source : str
        'OBS' ならパッケージ同梱の SILSO 年平均黒点数 (SN_Yearly.csv,
        1700-2024) を使う。それ以外は CSV ファイルパスとして読む
        (1行目はヘッダとしてスキップ)。
    time_col, ssn_col : int
        CSV の年・黒点数の列番号 (0始まり)。
    interval_days : float
        内挿間隔 [日]。シミュレーションの dtout と一致させること。

    Returns
    -------
    Observations
    """
    if source == 'OBS':
        with resources.as_file(_package_data_path('SN_Yearly.csv')) as p:
            data = np.genfromtxt(p, delimiter=',', skip_header=1)
    else:
        if not os.path.exists(source):
            raise FileNotFoundError(f"observation file not found: {source}")
        data = np.genfromtxt(source, delimiter=',', skip_header=1)
    years = data[:, time_col]
    ssn = data[:, ssn_col]
    return interpolate_to_interval(years, ssn, interval_days=interval_days)


def detect_minimum_years(source='OBS', order=27):
    """観測データの極小年リストを返す (旧 S2MFD.detect_minimum 相当)。"""
    return load_observations(source).detect_minima_years(order=order)


def load_initial_field():
    """スピンアップ用の初期磁場 (Jouve+2008 平衡場、128x128格子) を返す。

    Returns
    -------
    tuple of numpy.ndarray
        (Bph, Aph)。形状 (130, 130) = 128 + ゴーストセル2。
    """
    with resources.as_file(_package_data_path('Bpht_saved.npy')) as p:
        Bph = np.load(p)
    with resources.as_file(_package_data_path('Apht_saved.npy')) as p:
        Aph = np.load(p)
    return Bph, Aph
