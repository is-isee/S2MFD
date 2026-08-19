"""パラメタ推定の2段階ドライバ (旧 IDPA.py の argparse 版)。

使い方:
    python -m S2MFD.inference.cli --start-year 1944 --end-year 1996 \\
        --pop 30 --generations 50 --seed 42

対話モード (旧 IDPA.py 互換):
    python -m S2MFD.inference.cli --interactive

Stage 1 (u0) → Stage 2 (s0) の順に推定し、結果を
{output}/datas_{start}_{end}/data_{u0,s0}_{start}_{end}/ に保存する。
Stage 間の受け渡しは stage1_result.json で行う
(旧実装のようにパラメタファイルのソースコードを書き換えることはしない)。
"""
import argparse
import json
import logging
import os
import sys

import numpy as np

from .compare import compare_with_observation, plot_history
from .genetic import GAConfig, GeneticAlgorithm
from .observations import load_observations, load_initial_field
from .problem import DynamoProblem, TimeFunction, U0_KEYS, S0_KEYS

logger = logging.getLogger('S2MFD.inference')

YEAR_SECONDS = 365 * 24 * 3600


def build_parser():
    parser = argparse.ArgumentParser(
        prog='python -m S2MFD.inference.cli',
        description='GAによるダイナモモデルのパラメタ推定 (Shimizu & Hotta 2026)')
    parser.add_argument('--data', default='OBS',
                        help="観測データ: 'OBS' (同梱のSILSO年平均) または CSV パス")
    parser.add_argument('--start-year', type=float, default=None,
                        help='推定開始年 (極小年から選ぶこと)')
    parser.add_argument('--end-year', type=float, default=None,
                        help='推定終了年 (開始年+60年以下)')
    parser.add_argument('--pop', type=int, default=30, help='個体数 (論文: 30)')
    parser.add_argument('--generations', type=int, default=50,
                        help='最大世代数 (論文: 50)')
    parser.add_argument('--target-fitness', type=float, default=0.80,
                        help='目標適応度 (推奨: 0.80)')
    parser.add_argument('--output', default='results',
                        help='結果の保存先ルートディレクトリ')
    parser.add_argument('--parameter-file', default='parameters/inference.py',
                        help='シミュレーション設定ファイル')
    parser.add_argument('--stage', choices=['both', 'u0', 's0'], default='both',
                        help='実行する推定段階')
    parser.add_argument('--fitness', choices=['standard', 'period'],
                        default='standard',
                        help='適応度: standard=式3.1 / period=周期MSE (ダルトンミニマム用)')
    parser.add_argument('--seed', type=int, default=None, help='乱数シード')
    parser.add_argument('--workers', type=int, default=None,
                        help='並列ワーカー数 (既定: min(個体数, CPU数))')
    parser.add_argument('--spinup-years', type=float, default=80.0,
                        help='スピンアップ年数 (論文: 80)')
    parser.add_argument('--alpha', type=float, default=0.9,
                        help='適応度の相関/NMSE重み (論文: 0.9)')
    parser.add_argument('--clip-mutation', action='store_true',
                        help='変異を初期範囲にクリップする (実験的。既定は論文どおり非有界)')
    parser.add_argument('--list-minima', action='store_true',
                        help='観測データの極小年を表示して終了')
    parser.add_argument('--interactive', action='store_true',
                        help='対話モードで実行 (旧 IDPA.py 互換)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='デバッグログを表示')
    return parser


def ask(prompt, cast, default=None):
    while True:
        raw = input(prompt).strip()
        if raw == '' and default is not None:
            return default
        try:
            return cast(raw)
        except ValueError:
            print('数値を入力してください。')


def interactive_fill(args, obs):
    print('検出された極小年:', np.round(obs.detect_minima_years(), 2))
    args.start_year = ask('上記(極小年)より推定開始年を入力してください: ', float)
    args.end_year = ask(f'推定終了年を {args.start_year + 60} 年以下で入力してください: ', float)
    args.pop = ask('世代の中の個体数を入力してください (論文値30, コア数以下推奨): ', int, 30)
    args.generations = ask('最大世代数を入力 (論文値50): ', int, 50)
    args.target_fitness = ask('目標適応度を設定してください (0<Eva<1, 推奨0.80): ',
                              float, 0.80)
    return args


def run_stage(mode, args, obs, index_start, index_end, output_dir,
              stage1_params=None):
    """1段階分の推定を実行し、結果 dict を返す。"""
    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, 'stage_result.json')
    if os.path.exists(result_path):
        logger.info('%s の推定結果が %s に存在するためスキップします', mode, output_dir)
        with open(result_path) as f:
            return json.load(f)

    Bph0, Aph0 = load_initial_field()
    problem = DynamoProblem(
        parameter_file=args.parameter_file,
        obs_seconds=obs.seconds, obs_ssn=obs.ssn,
        index_start=index_start, index_end=index_end,
        mode=mode, stage1_params=stage1_params,
        initial_Bph=Bph0, initial_Aph=Aph0,
        spinup_years=args.spinup_years, alpha=args.alpha,
        fitness_kind=args.fitness,
    )
    config = GAConfig(
        threshold=args.target_fitness,
        max_generations=args.generations,
        clip_to_bounds=args.clip_mutation,
        max_workers=args.workers,
        seed=args.seed,
    )
    ga = GeneticAlgorithm(problem, problem.param_spec(),
                          population_size=args.pop, config=config)
    logger.info('=== Stage %s: GA開始 (個体数%d, 最大%d世代) ===',
                mode, args.pop, args.generations)
    ga_result = ga.run()
    logger.info('Stage %s 終了: fitness=%.6f (%d世代%s)',
                mode, ga_result.best_fitness, ga_result.generations_run,
                ', 中断' if ga_result.interrupted else '')

    # 最良個体で再シミュレーション (スナップショット保存)
    logger.info('最良個体で再シミュレーションを実行します')
    sn, ts, sim = problem.simulate(ga_result.best_params, save_dir=output_dir)

    # 振幅時系列を復元して比較プロット
    t_start = float(obs.seconds[index_start])
    t_span = float(obs.seconds[index_end]) - t_start
    best = ga_result.best_params
    if mode == 'u0':
        f_u = TimeFunction(*[best[k] for k in U0_KEYS], t_start, t_span)
        uu0_sim = np.array([f_u(t) for t in ts])
        so0_sim = np.full_like(uu0_sim, best['a0_s'])
    else:
        f_u = TimeFunction(*[stage1_params[k] for k in U0_KEYS], t_start, t_span)
        f_s = TimeFunction(*[best[k] for k in S0_KEYS], t_start, t_span)
        uu0_sim = np.array([f_u(t) for t in ts])
        so0_sim = np.array([f_s(t) for t in ts])

    scores = compare_with_observation(
        obs, index_start, index_end, sn, ts, uu0_sim, so0_sim,
        best, mode, output_dir, alpha=args.alpha)
    plot_history(ga_result, output_dir)

    # 結果の保存 (旧互換の parameters.txt + JSON)
    keys = list(problem.param_spec())
    np.savetxt(os.path.join(output_dir, 'parameters.txt'),
               np.array([best[k] for k in keys]).reshape(1, -1), fmt='%.6f')
    result = {
        'mode': mode,
        'params': {k: float(best[k]) for k in keys},
        'fitness': float(ga_result.best_fitness),
        'rerun_scores': scores,
        'generations_run': int(ga_result.generations_run),
        'interrupted': bool(ga_result.interrupted),
        'fitness_history': [float(v) for v in ga_result.fitness_history],
        'diversity_history': [float(v) for v in ga_result.diversity_history],
        'config': {
            'population': args.pop, 'max_generations': args.generations,
            'target_fitness': args.target_fitness, 'seed': args.seed,
            'alpha': args.alpha, 'spinup_years': args.spinup_years,
            'fitness_kind': args.fitness, 'data': args.data,
            'parameter_file': args.parameter_file,
        },
    }
    if stage1_params is not None:
        result['stage1_params'] = {k: float(v) for k, v in stage1_params.items()}
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    logger.info('結果を保存しました: %s', result_path)
    return result


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s %(name)s %(levelname)s: %(message)s')

    obs = load_observations(args.data)

    if args.list_minima:
        print('検出された極小年:', np.round(obs.detect_minima_years(), 2))
        return 0

    if args.interactive:
        args = interactive_fill(args, obs)
    if args.start_year is None or args.end_year is None:
        print('--start-year と --end-year を指定してください '
              '(極小年は --list-minima で確認できます)', file=sys.stderr)
        return 1

    span = args.end_year - args.start_year
    if span <= 0:
        print('終了年は開始年より大きい値を指定してください。', file=sys.stderr)
        return 1
    if span > 60:
        print('60年以上の推定は非推奨です。', file=sys.stderr)
        return 1

    index_start = obs.index_of_year(args.start_year)
    index_end = obs.index_of_year(args.end_year)
    st, en = int(args.start_year), int(args.end_year)
    root = os.path.join(args.output, f'datas_{st}_{en}')
    output_u0 = os.path.join(root, f'data_u0_{st}_{en}')
    output_s0 = os.path.join(root, f'data_s0_{st}_{en}')

    stage1 = None
    if args.stage in ('both', 'u0'):
        stage1 = run_stage('u0', args, obs, index_start, index_end, output_u0)

    if args.stage in ('both', 's0'):
        if stage1 is None:
            path = os.path.join(output_u0, 'stage_result.json')
            if not os.path.exists(path):
                print(f'Stage 1 の結果 {path} が見つかりません。'
                      '先に --stage u0 を実行してください。', file=sys.stderr)
                return 1
            with open(path) as f:
                stage1 = json.load(f)
        run_stage('s0', args, obs, index_start, index_end, output_s0,
                  stage1_params=stage1['params'])

    logger.info('完了。結果: %s', root)
    return 0


if __name__ == '__main__':
    sys.exit(main())
