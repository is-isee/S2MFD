"""フルスケール twin experiment (論文 Shimizu & Hotta 2026 の手順に準拠)。

手順 (論文 4.1 章):
1. 既知の真値 u0(t), s0(t) **両方を時間変化させて** 合成黒点数データ (GT) を作る
2. Step 1: u0(t) を推定 (s0 は定数 a0_s として同時に探索)
3. Step 2: Step 1 で推定した u0(t) を既知として s0(t) を推定
4. 真値との誤差を論文式 4.1 (時系列の絶対平均誤差) で評価

論文設定: 個体数30・最大50世代・80年スピンアップ・約5周期の窓。
実行例:
    OMP_NUM_THREADS=1 python doc/dev_records/verification/twin_experiment_full.py \\
        --workers 30 --output twin_full

結果は <output>/twin_full_result.json に保存される。
"""
import argparse
import json
import logging
import os
import time

import numpy as np

from S2MFD.inference.genetic import GAConfig, GeneticAlgorithm
from S2MFD.inference.metrics import correlation, nmse
from S2MFD.inference.observations import load_observations, load_initial_field
from S2MFD.inference.problem import DynamoProblem, TimeFunction, U0_KEYS, S0_KEYS

YEAR_SECONDS = 365 * 24 * 3600

# 真値 (論文表3.1 の探索範囲内に収まる、物理的に妥当な値)
TRUE_U0 = {'as_u': 750.0, 'ae_u': 650.0, 'a1_u': 60.0, 'a2_u': -40.0, 'a4_u': 25.0}
TRUE_S0 = {'as_s': 55.0, 'ae_s': 45.0, 'a1_s': 6.0, 'a2_s': -4.0, 'a4_s': 3.0}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--start-year', type=float, default=1944.45)
    p.add_argument('--end-year', type=float, default=1996.5)
    p.add_argument('--pop', type=int, default=30)
    p.add_argument('--generations', type=int, default=50)
    p.add_argument('--threshold', type=float, default=0.895,
                   help='twin では理論最大 0.9。論文表4.1 相当 (r=0.995) は 0.8945')
    p.add_argument('--spinup-years', type=float, default=80.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--workers', type=int, default=30)
    p.add_argument('--output', default='twin_full')
    p.add_argument('--parameter-file', default='parameters/inference.py')
    return p.parse_args()


def series_error(true_params, est_params, keys, t_start, t_span, n=1000):
    """論文式 4.1: (1/n) Σ |x_GT(t) - x_est(t)| / |x_GT(t)|"""
    t = np.linspace(t_start, t_start + t_span, n)
    f_true = TimeFunction(*[true_params[k] for k in keys], t_start, t_span)
    f_est = TimeFunction(*[est_params[k] for k in keys], t_start, t_span)
    x_true = np.array([f_true(ti) for ti in t])
    x_est = np.array([f_est(ti) for ti in t])
    return float(np.mean(np.abs(x_true - x_est) / np.abs(x_true)))


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s')
    log = logging.getLogger('twin_full')

    obs = load_observations('OBS')
    Bph0, Aph0 = load_initial_field()
    i0 = obs.index_of_year(args.start_year)
    i1 = obs.index_of_year(args.end_year)
    t_start = float(obs.seconds[i0])
    t_span = float(obs.seconds[i1]) - t_start
    log.info('window: index %d..%d (%.2f - %.2f yr, %.2f yr, %d points)',
             i0, i1, obs.years[i0], obs.years[i1],
             obs.years[i1] - obs.years[i0], i1 - i0 + 1)
    log.info('true u0: %s', TRUE_U0)
    log.info('true s0: %s', TRUE_S0)

    def make_problem(mode, obs_ssn, stage1=None):
        return DynamoProblem(
            parameter_file=args.parameter_file,
            obs_seconds=obs.seconds, obs_ssn=obs_ssn,
            index_start=i0, index_end=i1,
            mode=mode, stage1_params=stage1,
            initial_Bph=Bph0, initial_Aph=Aph0,
            spinup_years=args.spinup_years, until_minimum=True)

    results = {
        'setup': {
            'start_year': obs.years[i0], 'end_year': obs.years[i1],
            'index_start': i0, 'index_end': i1,
            'pop': args.pop, 'generations': args.generations,
            'threshold': args.threshold, 'spinup_years': args.spinup_years,
            'seed': args.seed, 'true_u0': TRUE_U0, 'true_s0': TRUE_S0,
        }
    }

    # ---------------- Ground Truth の生成 ----------------
    # u0(t), s0(t) の両方を時間変化させた「真の太陽」を作る
    log.info('=== Ground Truth の生成 (u0, s0 とも時間変化) ===')
    t0 = time.time()
    gt_problem = make_problem('s0', obs.ssn, stage1=TRUE_U0)
    sn_gt, ts_gt, _ = gt_problem.simulate(TRUE_S0)
    log.info('GT 完了 (%.0f s): SSN range %.1f - %.1f',
             time.time() - t0, sn_gt.min(), sn_gt.max())

    obs_gt = obs.ssn.copy()
    obs_gt[i0:i1 + 1] = sn_gt
    np.savez(os.path.join(args.output, 'ground_truth.npz'),
             sn=sn_gt, ts=ts_gt, index_start=i0, index_end=i1)

    # ---------------- Step 1: u0(t) の推定 ----------------
    log.info('=== Step 1: u0(t) 推定 (pop=%d, max_gen=%d, workers=%d) ===',
             args.pop, args.generations, args.workers)
    step1 = make_problem('u0', obs_gt)
    ga1 = GeneticAlgorithm(
        step1, step1.param_spec(), population_size=args.pop,
        config=GAConfig(threshold=args.threshold,
                        max_generations=args.generations,
                        max_workers=args.workers, seed=args.seed))
    t0 = time.time()
    r1 = ga1.run()
    elapsed1 = time.time() - t0
    err_u_step1 = series_error(TRUE_U0, r1.best_params, U0_KEYS, t_start, t_span)
    log.info('Step 1 完了 (%.0f s, %d世代): fitness=%.6f, u0(t)誤差=%.2f%%',
             elapsed1, r1.generations_run, r1.best_fitness, err_u_step1 * 100)

    sn1, ts1, _ = step1.simulate(r1.best_params,
                                 save_dir=os.path.join(args.output, 'step1'))
    r_sunspot_1 = correlation(sn_gt, sn1)
    e_sunspot_1 = nmse(sn_gt, sn1)
    # Step 1 時点の s0 は定数 a0_s なので、真の s0(t) との誤差も参考記録
    log.info('Step 1: r_sunspot=%.4f, e_sunspot=%.5f, a0_s=%.2f',
             r_sunspot_1, e_sunspot_1, r1.best_params['a0_s'])

    results['step1'] = {
        'params': {k: float(v) for k, v in r1.best_params.items()},
        'fitness': float(r1.best_fitness),
        'generations_run': int(r1.generations_run),
        'elapsed_sec': elapsed1,
        'u0_series_error': err_u_step1,
        'r_sunspot': r_sunspot_1, 'e_sunspot': e_sunspot_1,
        'fitness_history': [float(v) for v in r1.fitness_history],
        'diversity_history': [float(v) for v in r1.diversity_history],
    }
    with open(os.path.join(args.output, 'twin_full_result.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # ---------------- Step 2: s0(t) の推定 ----------------
    # Step 1 で推定した u0(t) を既知として使う (論文の2段階推定)
    stage1_u0 = {k: r1.best_params[k] for k in U0_KEYS}
    log.info('=== Step 2: s0(t) 推定 (Step 1 の u0 を既知として使用) ===')
    step2 = make_problem('s0', obs_gt, stage1=stage1_u0)
    ga2 = GeneticAlgorithm(
        step2, step2.param_spec(), population_size=args.pop,
        config=GAConfig(threshold=args.threshold,
                        max_generations=args.generations,
                        max_workers=args.workers, seed=args.seed))
    t0 = time.time()
    r2 = ga2.run()
    elapsed2 = time.time() - t0
    err_s_step2 = series_error(TRUE_S0, r2.best_params, S0_KEYS, t_start, t_span)
    log.info('Step 2 完了 (%.0f s, %d世代): fitness=%.6f, s0(t)誤差=%.2f%%',
             elapsed2, r2.generations_run, r2.best_fitness, err_s_step2 * 100)

    sn2, ts2, _ = step2.simulate(r2.best_params,
                                 save_dir=os.path.join(args.output, 'step2'))
    r_sunspot_2 = correlation(sn_gt, sn2)
    e_sunspot_2 = nmse(sn_gt, sn2)
    log.info('Step 2: r_sunspot=%.4f, e_sunspot=%.5f', r_sunspot_2, e_sunspot_2)

    results['step2'] = {
        'params': {k: float(v) for k, v in r2.best_params.items()},
        'stage1_u0': {k: float(v) for k, v in stage1_u0.items()},
        'fitness': float(r2.best_fitness),
        'generations_run': int(r2.generations_run),
        'elapsed_sec': elapsed2,
        's0_series_error': err_s_step2,
        'u0_series_error': err_u_step1,  # Step2 でも u0 は Step1 のまま
        'r_sunspot': r_sunspot_2, 'e_sunspot': e_sunspot_2,
        'fitness_history': [float(v) for v in r2.fitness_history],
        'diversity_history': [float(v) for v in r2.diversity_history],
    }

    with open(os.path.join(args.output, 'twin_full_result.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # ---------------- サマリ (論文表4.1 と同じ並び) ----------------
    log.info('==================== 結果サマリ ====================')
    log.info('           r_sunspot  e_sunspot  u0誤差   s0誤差')
    log.info('Step 1     %.4f     %.5f    %.2f%%    (定数)',
             r_sunspot_1, e_sunspot_1, err_u_step1 * 100)
    log.info('Step 2     %.4f     %.5f    %.2f%%    %.2f%%',
             r_sunspot_2, e_sunspot_2, err_u_step1 * 100, err_s_step2 * 100)
    log.info('論文表4.1 No.1: 0.995  0.00514  2.87%%  5.25%%')
    log.info('結果を保存しました: %s/twin_full_result.json', args.output)


if __name__ == '__main__':
    main()
