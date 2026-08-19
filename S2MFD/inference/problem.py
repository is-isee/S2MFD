"""ダイナモシミュレーションによる個体評価 (GA と Simulation の接続)。

旧 IDPA の DefunctionProblem + run_defunction_simulation の再設計。

個体評価の流れ (Shimizu & Hotta 2026, 2.7章):

1. Jouve+2008 の平衡場から、ゲノムの開始振幅 (定数) で 80 年 + 黒点数
   極小期までスピンアップ
2. 時刻を観測窓の開始に整列し、時間変化関数 (線形 + sin級数) をフックとして
   取り付けて観測窓を積分
3. 40日間隔の黒点数プロキシ時系列を観測と比較して適応度を返す

旧実装との差異:

- 個体評価はディスク出力なし (save_dir=None)。並列実行時の出力ディレクトリ
  競合が原因ごと消える。最良個体の最終ランのみ save_dir を指定して
  スナップショットを保存する。
- 数値的に不安定な個体 (NaN 発散) は例外ではなく適応度 -inf として扱う。
"""
import logging
import os
from dataclasses import dataclass, field

import numpy as np

import S2MFD
from S2MFD.tools import sunspot_proxy
from . import metrics

__all__ = ['DynamoProblem', 'TimeFunction', 'linear_plus_sines',
           'U0_SPEC', 'S0_SPEC', 'U0_KEYS', 'S0_KEYS']

logger = logging.getLogger(__name__)

YEAR_SECONDS = 365 * 24 * 3600

# 初期集団の一様分布範囲 (論文表 3.1)
U0_SPEC = {
    'as_u': (500.0, 900.0),
    'ae_u': (500.0, 900.0),
    'a1_u': (-150.0, 150.0),
    'a2_u': (-150.0, 150.0),
    'a4_u': (-150.0, 150.0),
    'a0_s': (40.0, 65.0),   # u0 推定中は定数として扱う α 振幅
}
S0_SPEC = {
    'as_s': (40.0, 65.0),
    'ae_s': (40.0, 65.0),
    'a1_s': (-15.0, 15.0),
    'a2_s': (-15.0, 15.0),
    'a4_s': (-15.0, 15.0),
}

U0_KEYS = ('as_u', 'ae_u', 'a1_u', 'a2_u', 'a4_u')
S0_KEYS = ('as_s', 'ae_s', 'a1_s', 'a2_s', 'a4_s')


def linear_plus_sines(t, a_start, a_end, a1, a2, a4, t_span):
    """時間変化関数: 線形 + sin級数 (論文式 2.10, lmax=4, l=3 除外)。

    Parameters
    ----------
    t : float
        窓開始からの経過時間 [s] (0 <= t <= t_span)
    a_start, a_end : float
        窓の始点・終点での振幅 (線形成分)
    a1, a2, a4 : float
        sin 級数の係数
    t_span : float
        窓の長さ [s]。omega = 2*pi / (2 * t_span) (窓の2倍を基本周期とする)
    """
    omega = 2 * np.pi / (t_span * 2)
    lin = (a_start * (t_span - t) + a_end * t) / t_span
    return (lin
            + a1 * np.sin(1.0 * omega * t)
            + a2 * np.sin(2.0 * omega * t)
            + a4 * np.sin(4.0 * omega * t))


class TimeFunction:
    """絶対時刻 [s] を受けて振幅を返す呼び出し可能オブジェクト。

    cfg.uu0_of_time / cfg.so0_of_time にそのまま渡せる。
    (クロージャと違い picklable なので、必要ならプロセス間も運べる)
    """

    def __init__(self, a_start, a_end, a1, a2, a4, t_start, t_span):
        self.a_start = float(a_start)
        self.a_end = float(a_end)
        self.a1 = float(a1)
        self.a2 = float(a2)
        self.a4 = float(a4)
        self.t_start = float(t_start)
        self.t_span = float(t_span)

    def __call__(self, t):
        return linear_plus_sines(t - self.t_start,
                                 self.a_start, self.a_end,
                                 self.a1, self.a2, self.a4, self.t_span)


@dataclass
class DynamoProblem:
    """GA の評価器: evaluator(params) -> fitness。

    Attributes
    ----------
    parameter_file : str
        シミュレーション設定 (S2MFD.Cfg に渡すパス)
    obs_seconds, obs_ssn : numpy.ndarray
        40日間隔に内挿済みの観測時系列 (observations.Observations の
        seconds / ssn)
    index_start, index_end : int
        推定窓の観測インデックス範囲 [index_start, index_end]
    mode : str
        'u0' (Step 1: 子午面流) または 's0' (Step 2: α効果)
    stage1_params : dict, optional
        mode='s0' のとき必須。Step 1 で推定した u0 係数
        {'as_u', 'ae_u', 'a1_u', 'a2_u', 'a4_u'}
    initial_Bph, initial_Aph : numpy.ndarray
        スピンアップの初期磁場 (observations.load_initial_field() 等)
    spinup_years : float
        スピンアップの最低年数 (論文では 80 年)
    alpha : float
        適応度の重み (論文では 0.9)
    """
    parameter_file: str
    obs_seconds: np.ndarray
    obs_ssn: np.ndarray
    index_start: int
    index_end: int
    mode: str
    initial_Bph: np.ndarray
    initial_Aph: np.ndarray
    stage1_params: dict = None
    spinup_years: float = 80.0
    until_minimum: bool = True  # スピンアップ後に黒点数極小まで継続するか
    alpha: float = 0.9
    fitness_kind: str = 'standard'  # 'standard' (式3.1) or 'period' (式4.2)

    def __post_init__(self):
        if self.mode not in ('u0', 's0'):
            raise ValueError(f"mode must be 'u0' or 's0', got {self.mode!r}")
        if self.mode == 's0' and self.stage1_params is None:
            raise ValueError("mode='s0' requires stage1_params (Step 1 の結果)")

    # ------------------------------------------------------------------ #
    @property
    def window_length(self):
        return self.index_end - self.index_start + 1

    @property
    def obs_window(self):
        return self.obs_ssn[self.index_start:self.index_end + 1]

    def param_spec(self):
        return dict(U0_SPEC) if self.mode == 'u0' else dict(S0_SPEC)

    # ------------------------------------------------------------------ #
    def simulate(self, params, save_dir=None):
        """1個体分のシミュレーションを実行する。

        Parameters
        ----------
        params : dict
            ゲノム (U0_SPEC / S0_SPEC のキー)
        save_dir : str, optional
            指定すると観測窓のスナップショット一式を保存する
            (最良個体の最終ラン用)。None なら完全にディスク出力なし。

        Returns
        -------
        tuple
            (sn_sim, ts_sim, sim) — 黒点数プロキシ時系列 (長さ
            window_length)、対応する時刻 [s]、Simulation オブジェクト。
        """
        t_start = float(self.obs_seconds[self.index_start])
        t_end = float(self.obs_seconds[self.index_end])
        t_span = t_end - t_start

        cfg = S2MFD.Cfg(self.parameter_file)
        cfg.verbose = False
        # スピンアップは開始振幅の定数で行う
        if self.mode == 'u0':
            cfg.uu0 = float(params['as_u'])
            cfg.so0 = float(params['a0_s'])
        else:
            cfg.uu0 = float(self.stage1_params['as_u'])
            cfg.so0 = float(params['as_s'])

        if save_dir is not None:
            cfg.datadir = save_dir
            cfg.cont_flag = False

        sim = S2MFD.Simulation(cfg)
        if save_dir is not None:
            sim.initialize_simulation()
        sim.cfl_condition()
        sim.set_field(self.initial_Bph, self.initial_Aph, time=0.0)
        sim.spin_up(self.spinup_years * YEAR_SECONDS,
                    until_minimum=self.until_minimum)

        # 観測窓の開始に整列
        sim.time = t_start
        sim.n = 0
        sim.nd = self.index_start

        # 時間変化関数の取り付け
        if self.mode == 'u0':
            cfg.uu0_of_time = TimeFunction(
                params['as_u'], params['ae_u'],
                params['a1_u'], params['a2_u'], params['a4_u'],
                t_start, t_span)
        else:
            s1 = self.stage1_params
            cfg.uu0_of_time = TimeFunction(
                s1['as_u'], s1['ae_u'], s1['a1_u'], s1['a2_u'], s1['a4_u'],
                t_start, t_span)
            cfg.so0_of_time = TimeFunction(
                params['as_s'], params['ae_s'],
                params['a1_s'], params['a2_s'], params['a4_s'],
                t_start, t_span)

        sn = [sunspot_proxy(sim.Bph, sim.grid, cfg)]
        ts = [sim.time]
        if save_dir is not None:
            sim.save()

        def record(s):
            sn.append(sunspot_proxy(s.Bph, s.grid, s.cfg))
            ts.append(s.time)

        sim.run_window(t_end, on_output=record,
                       save_output=(save_dir is not None))

        sn = np.array(sn)
        ts = np.array(ts)
        L = self.window_length
        if len(sn) != L:
            logger.warning("simulated series length %d != window length %d "
                           "(adjusting)", len(sn), L)
            if len(sn) > L:
                sn, ts = sn[:L], ts[:L]
            else:
                sn = np.pad(sn, (0, L - len(sn)), mode='edge')
                ts = np.pad(ts, (0, L - len(ts)), mode='edge')
        return sn, ts, sim

    # ------------------------------------------------------------------ #
    def __call__(self, params):
        """適応度を返す。

        fitness_kind='standard': F = alpha*相関 - (1-alpha)*NMSE (論文式 3.1)
        fitness_kind='period':   F = -周期MSE (ダルトンミニマム用、論文式 4.2)

        数値的に破綻した個体 (NaN 発散など) は -inf を返す。
        """
        try:
            sn, ts, _ = self.simulate(params)
        except (RuntimeError, FloatingPointError) as exc:
            logger.warning("individual failed (%s); fitness = -inf", exc)
            return -np.inf
        if self.fitness_kind == 'period':
            t_obs = self.obs_seconds[self.index_start:self.index_end + 1]
            return -metrics.period_mse(self.obs_window, sn, t_obs, ts)
        return metrics.fitness(self.obs_window, sn, alpha=self.alpha)
