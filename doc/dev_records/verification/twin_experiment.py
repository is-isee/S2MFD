"""縮小版 twin experiment: 既知パラメタで合成した観測から GA がパラメタを回収できるか。

論文の設定 (80年スピンアップ、55年窓、個体数30×50世代) の縮小版:
- スピンアップ 10年 + 極小期まで
- 窓 22年 (約2周期)
- Step1: 個体数12×最大8世代 / Step2: 同様
計算時間の都合による縮小であり、手法の妥当性確認 (パイプラインの正しさ +
GA が真値方向に収束すること) を目的とする。論文水準の精度検証は
フルスケール設定で別途行うこと。
"""
import json
import logging
import time

import numpy as np

from S2MFD.inference.genetic import GAConfig, GeneticAlgorithm
from S2MFD.inference.observations import load_observations, load_initial_field
from S2MFD.inference.problem import DynamoProblem, U0_KEYS, S0_KEYS

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')
log = logging.getLogger('twin')

SPINUP_YEARS = 10.0
SEED = 42
POP = 12
GENS = 8

obs = load_observations('OBS')
B0, A0 = load_initial_field()
i0 = obs.index_of_year(1944)
i1 = obs.index_of_year(1966)
log.info('window: %d..%d (%.1f - %.1f yr, %d points)',
         i0, i1, obs.years[i0], obs.years[i1], i1 - i0 + 1)

TRUE_U = {'as_u': 750.0, 'ae_u': 650.0, 'a1_u': 60.0, 'a2_u': -40.0,
          'a4_u': 25.0, 'a0_s': 50.0}
TRUE_S = {'as_s': 55.0, 'ae_s': 45.0, 'a1_s': 6.0, 'a2_s': -4.0, 'a4_s': 3.0}


def make_problem(mode, obs_ssn, stage1=None):
    return DynamoProblem(
        'parameters/inference.py', obs.seconds, obs_ssn,
        i0, i1, mode, B0, A0, stage1_params=stage1,
        spinup_years=SPINUP_YEARS, until_minimum=True)


def param_errors(true, est, keys):
    return {k: abs(est[k] - true[k]) / max(abs(true[k]), 1e-12) for k in keys}


results = {}

# ============ GT-1: u0 変動 + s0 定数 → Step1 で回収 ============
log.info('=== GT-1 生成 (u0変動, s0定数) ===')
gt1 = make_problem('u0', obs.ssn)
t0 = time.time()
sn_true1, ts_true1, _ = gt1.simulate(TRUE_U)
log.info('GT-1 完了 (%.0fs)', time.time() - t0)

obs_syn1 = obs.ssn.copy()
obs_syn1[i0:i1 + 1] = sn_true1

step1 = make_problem('u0', obs_syn1)
f_true = step1(TRUE_U)
log.info('sanity: fitness(真値) = %.6f (期待値 0.9)', f_true)
results['sanity_fitness_true_u0'] = f_true

log.info('=== Step1 GA (pop=%d, gens=%d, seed=%d) ===', POP, GENS, SEED)
ga1 = GeneticAlgorithm(step1, step1.param_spec(), population_size=POP,
                       config=GAConfig(threshold=0.895, max_generations=GENS,
                                       seed=SEED))
t0 = time.time()
r1 = ga1.run()
log.info('Step1 完了 (%.0fs): fitness=%.6f', time.time() - t0, r1.best_fitness)
err1 = param_errors(TRUE_U, r1.best_params, list(TRUE_U))
log.info('Step1 真値との相対誤差: %s',
         {k: f'{v:.1%}' for k, v in err1.items()})
results['step1'] = {'fitness': r1.best_fitness, 'params': r1.best_params,
                    'errors': err1, 'history': list(r1.fitness_history),
                    'generations': r1.generations_run}

# ============ GT-2: u0既知 + s0 変動 → Step2 で回収 ============
log.info('=== GT-2 生成 (u0既知, s0変動) ===')
gt2 = make_problem('s0', obs.ssn, stage1=TRUE_U)
t0 = time.time()
sn_true2, ts_true2, _ = gt2.simulate(TRUE_S)
log.info('GT-2 完了 (%.0fs)', time.time() - t0)

obs_syn2 = obs.ssn.copy()
obs_syn2[i0:i1 + 1] = sn_true2

step2 = make_problem('s0', obs_syn2, stage1=TRUE_U)
f_true2 = step2(TRUE_S)
log.info('sanity: fitness(真値) = %.6f (期待値 0.9)', f_true2)
results['sanity_fitness_true_s0'] = f_true2

log.info('=== Step2 GA (pop=%d, gens=%d, seed=%d) ===', POP, GENS, SEED)
ga2 = GeneticAlgorithm(step2, step2.param_spec(), population_size=POP,
                       config=GAConfig(threshold=0.895, max_generations=GENS,
                                       seed=SEED))
t0 = time.time()
r2 = ga2.run()
log.info('Step2 完了 (%.0fs): fitness=%.6f', time.time() - t0, r2.best_fitness)
err2 = param_errors(TRUE_S, r2.best_params, list(TRUE_S))
log.info('Step2 真値との相対誤差: %s',
         {k: f'{v:.1%}' for k, v in err2.items()})
results['step2'] = {'fitness': r2.best_fitness, 'params': r2.best_params,
                    'errors': err2, 'history': list(r2.fitness_history),
                    'generations': r2.generations_run}

with open('twin_result.json', 'w') as f:
    json.dump(results, f, indent=2, default=float)
log.info('twin_result.json に保存しました')
