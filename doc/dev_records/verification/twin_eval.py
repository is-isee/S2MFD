"""twin experiment の結果を論文の基準 (時系列の絶対平均誤差、式4.1) で評価する。"""
import json

import numpy as np

from S2MFD.inference.observations import load_observations
from S2MFD.inference.problem import TimeFunction, U0_KEYS, S0_KEYS

YEAR_SECONDS = 365 * 24 * 3600

obs = load_observations('OBS')
i0 = obs.index_of_year(1944)
i1 = obs.index_of_year(1966)
t_start = float(obs.seconds[i0])
t_span = float(obs.seconds[i1]) - t_start
t = np.linspace(t_start, t_start + t_span, 500)

TRUE_U = {'as_u': 750.0, 'ae_u': 650.0, 'a1_u': 60.0, 'a2_u': -40.0,
          'a4_u': 25.0, 'a0_s': 50.0}
TRUE_S = {'as_s': 55.0, 'ae_s': 45.0, 'a1_s': 6.0, 'a2_s': -4.0, 'a4_s': 3.0}

with open('twin_result.json') as f:
    res = json.load(f)


def series_error(true_p, est_p, keys):
    """論文式4.1: (1/n) Σ |x_GT(t) - x_est(t)| / x_GT(t)"""
    f_true = TimeFunction(*[true_p[k] for k in keys], t_start, t_span)
    f_est = TimeFunction(*[est_p[k] for k in keys], t_start, t_span)
    x_true = np.array([f_true(ti) for ti in t])
    x_est = np.array([f_est(ti) for ti in t])
    return float(np.mean(np.abs(x_true - x_est) / np.abs(x_true)))


e_u = series_error(TRUE_U, res['step1']['params'], U0_KEYS)
print(f"Step1: fitness={res['step1']['fitness']:.6f} "
      f"(最大0.9), u0(t) 時系列誤差 = {e_u:.2%}")
a0_err = abs(res['step1']['params']['a0_s'] - TRUE_U['a0_s']) / TRUE_U['a0_s']
print(f"       a0_s (定数α) 誤差 = {a0_err:.2%}")

e_s = series_error(TRUE_S, res['step2']['params'], S0_KEYS)
print(f"Step2: fitness={res['step2']['fitness']:.6f} "
      f"(最大0.9), s0(t) 時系列誤差 = {e_s:.2%}")
print(f"sanity(真値の適応度): u0={res['sanity_fitness_true_u0']:.6f}, "
      f"s0={res['sanity_fitness_true_s0']:.6f} (期待値 0.9)")
