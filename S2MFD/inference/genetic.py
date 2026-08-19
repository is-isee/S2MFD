"""パラメタ非依存の遺伝的アルゴリズム (GA)。

旧 IDPA の GA_for_u0.py / GA_for_s0.py (98% 重複した2ファイル) を、
推定パラメタの仕様 (param_spec) を引数で受ける1つの実装に統合したもの。

アルゴリズムは Shimizu & Hotta (2026) 3章に準拠:

- 選択: ASP (Adaptive Selection Pressure) トーナメント。
  適応度変化率と多様性に応じてトーナメントサイズを 3/7/5 に切替
- 交叉: SBX (Simulated Binary Crossover), eta=2, 交叉確率 0.8
- 変異: ガウス変異 (sigma = 0.1|x|), 変異確率 0.3
- エリート保存

旧実装との意図的な差異 (doc/dev_records/2026-08-19_idpa_analysis.md 参照):

- トーナメントサイズは個体数比 (len//10 等) ではなく論文どおりの固定値
  3/7/5 (個体数30では両者は一致する)。少数個体でも壊れないようガード付き。
- エリートは固定スロット3ではなくランダムな1個体を置換する。
- 乱数は numpy Generator に統一し、seed 指定で再現可能。
- 個体評価はディスク出力なしで行い、並列実行時の出力ディレクトリ競合を
  解消した (評価器側の設計)。
"""
import logging
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np

__all__ = ['GAConfig', 'Chromosome', 'GAResult', 'GeneticAlgorithm']

logger = logging.getLogger(__name__)


@dataclass
class GAConfig:
    """GA のハイパーパラメタ。既定値は論文 (Shimizu & Hotta 2026) に準拠。"""
    threshold: float = 0.80          # 目標適応度 (到達で終了)
    max_generations: int = 50
    mutation_probability: float = 0.3
    crossover_probability: float = 0.8
    eta_sbx: float = 2.0             # SBX 分布指数
    mutation_sigma_frac: float = 0.1  # ガウス変異の sigma = frac * |x|
    epsi_fit: float = 0.005          # ASP: 適応度停滞の閾値
    epsi_div: float = 0.10           # ASP: 多様性喪失の閾値
    tournament_low: int = 3          # 停滞かつ多様性低 → 選択圧(低)
    tournament_high: int = 7         # 停滞かつ多様性高 → 選択圧(大)
    tournament_mid: int = 5          # それ以外 → 選択圧(中)
    # --- オプション (既定は論文実装のまま = 無効) ---
    clip_to_bounds: bool = False     # 変異・交叉後に初期範囲へクリップ
    mutation_sigma_min: float = 0.0  # sigma の下限 (0付近での凍結回避)
    # --- 実行制御 ---
    max_workers: int = None          # None なら min(個体数, CPU数)
    seed: int = None


@dataclass
class Chromosome:
    params: dict
    fitness: float = None


@dataclass
class GAResult:
    best_params: dict
    best_fitness: float
    fitness_history: np.ndarray      # 世代ごとの最良適応度
    diversity_history: np.ndarray    # 世代ごとの多様性
    generations_run: int
    interrupted: bool = False
    population: list = field(default_factory=list)  # 最終世代 (Chromosome)


def _evaluate_one(args):
    evaluator, params = args
    return evaluator(params)


class GeneticAlgorithm:
    """遺伝的アルゴリズム本体。

    Parameters
    ----------
    evaluator : callable
        evaluator(params: dict) -> float (適応度)。並列実行のため
        picklable であること (クロージャ不可。dataclass 等を推奨)。
    param_spec : dict[str, tuple[float, float]]
        推定パラメタ名 → 初期集団の一様分布範囲 (min, max)。
        例 (u0推定): {'as_u': (500, 900), ..., 'a0_s': (40, 65)}
    population_size : int
        個体数 (論文では 30)。
    config : GAConfig, optional
    """

    def __init__(self, evaluator, param_spec, population_size,
                 config=None, initial_population=None):
        self.evaluator = evaluator
        self.param_spec = dict(param_spec)
        self.config = config or GAConfig()
        self.rng = np.random.default_rng(self.config.seed)

        if initial_population is not None:
            self.population = [Chromosome(params=dict(p)) for p in initial_population]
        else:
            self.population = [self._make_random_chromosome()
                               for _ in range(population_size)]
        if len(self.population) < 2:
            raise ValueError("population_size must be >= 2")

        self.fitness_history = np.zeros(self.config.max_generations)
        self.diversity_history = np.zeros(self.config.max_generations)
        self.generation_idx = 0

    # ------------------------------------------------------------------ #
    # 個体生成
    def _make_random_chromosome(self):
        params = {name: self.rng.uniform(lo, hi)
                  for name, (lo, hi) in self.param_spec.items()}
        return Chromosome(params=params)

    # ------------------------------------------------------------------ #
    # 評価
    def _evaluate_population(self):
        """未評価の個体を並列に評価する。"""
        todo = [c for c in self.population if c.fitness is None]
        if not todo:
            return
        max_workers = self.config.max_workers or min(len(todo), _cpu_count())
        args = [(self.evaluator, c.params) for c in todo]
        if max_workers <= 1:
            results = [_evaluate_one(a) for a in args]
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(_evaluate_one, args))
        for chrom, fit in zip(todo, results):
            chrom.fitness = float(fit)

    def _best(self):
        return max(self.population, key=lambda c: c.fitness)

    # ------------------------------------------------------------------ #
    # 多様性 (論文式 3.7, 3.8)
    def diversity(self):
        """個体群の多様性: パラメタ毎に min-max 正規化した標準偏差の平均。"""
        stds = []
        for name in self.param_spec:
            values = np.array([c.params[name] for c in self.population])
            vmin, vmax = values.min(), values.max()
            if vmax - vmin > 0:
                values = (values - vmin) / (vmax - vmin)
            stds.append(np.std(values))
        return float(np.mean(stds))

    # ------------------------------------------------------------------ #
    # 選択 (ASPトーナメント、論文式 3.9)
    def _tournament_size(self):
        if self.generation_idx > 0:
            prev = self.fitness_history[self.generation_idx - 1]
            cur = self.fitness_history[self.generation_idx]
            fitness_delta = abs(cur - prev) / abs(prev) if prev != 0 else 10.0
        else:
            fitness_delta = 10.0
        div = self.diversity()
        cfg = self.config
        if fitness_delta < cfg.epsi_fit:
            if div < cfg.epsi_div:
                k, label = cfg.tournament_low, 'low'    # 局所解の可能性 → 圧を下げる
            else:
                k, label = cfg.tournament_high, 'high'  # 収束段階 → 圧を上げる
        else:
            k, label = cfg.tournament_mid, 'mid'        # 探索段階
        k = max(2, min(k, len(self.population)))
        logger.debug("ASP tournament: size=%d (%s) dF=%.4g D=%.4g",
                     k, label, fitness_delta, div)
        return k

    def _select_parents(self):
        k = self._tournament_size()
        idx = self.rng.integers(0, len(self.population), size=k)  # 重複許容
        participants = [self.population[i] for i in idx]
        participants.sort(key=lambda c: c.fitness, reverse=True)
        return participants[0], participants[1]

    # ------------------------------------------------------------------ #
    # 交叉 (SBX、論文式 3.10, 3.11)
    def _sbx_crossover(self, parent_1, parent_2):
        child_1 = deepcopy(parent_1)
        child_2 = deepcopy(parent_2)
        eta = self.config.eta_sbx
        for name in self.param_spec:
            if self.rng.random() > 0.5:  # 遺伝子ごとに交叉するか決定
                u = self.rng.random()
                if u <= 0.5:
                    beta = (2.0 * u) ** (1.0 / (eta + 1.0))
                else:
                    beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))
                p1 = parent_1.params[name]
                p2 = parent_2.params[name]
                child_1.params[name] = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
                child_2.params[name] = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
        child_1.fitness = None
        child_2.fitness = None
        return child_1, child_2

    # ------------------------------------------------------------------ #
    # 変異 (ガウス変異)
    def _gaussian_mutation(self, chromosome):
        name = list(self.param_spec)[self.rng.integers(len(self.param_spec))]
        before = chromosome.params[name]
        sigma = max(self.config.mutation_sigma_frac * abs(before),
                    self.config.mutation_sigma_min)
        chromosome.params[name] = float(self.rng.normal(loc=before, scale=sigma))
        chromosome.fitness = None
        logger.debug("gaussian_mutation: %s %.6g -> %.6g",
                     name, before, chromosome.params[name])

    def _clip(self, chromosome):
        for name, (lo, hi) in self.param_spec.items():
            chromosome.params[name] = float(np.clip(chromosome.params[name], lo, hi))

    # ------------------------------------------------------------------ #
    # 世代交代
    def _next_generation(self, elite):
        new_population = []
        while len(new_population) < len(self.population):
            parent_1, parent_2 = self._select_parents()
            if self.rng.random() < self.config.crossover_probability:
                child_1, child_2 = self._sbx_crossover(parent_1, parent_2)
            else:
                child_1, child_2 = deepcopy(parent_1), deepcopy(parent_2)
            if self.rng.random() < self.config.mutation_probability:
                self._gaussian_mutation(child_1)
                self._gaussian_mutation(child_2)
            if self.config.clip_to_bounds:
                self._clip(child_1)
                self._clip(child_2)
            new_population.extend([child_1, child_2])

        if len(new_population) > len(self.population):
            del new_population[0]

        # エリート保存: ランダムな1個体を最良個体で置換
        slot = int(self.rng.integers(len(new_population)))
        new_population[slot] = deepcopy(elite)
        self.population = new_population

    # ------------------------------------------------------------------ #
    def run(self):
        """GA を実行して最良個体を返す。

        Returns
        -------
        GAResult
            threshold 到達または max_generations 完了時点の結果。
            KeyboardInterrupt 時は interrupted=True で途中結果を返す。
        """
        cfg = self.config
        best = None
        interrupted = False
        gen = 0
        try:
            self._evaluate_population()
            best = deepcopy(self._best())
            for gen in range(cfg.max_generations):
                self.generation_idx = gen
                self.fitness_history[gen] = best.fitness
                self.diversity_history[gen] = self.diversity()
                logger.info("generation %d: best fitness=%.6f diversity=%.4f",
                            gen, best.fitness, self.diversity_history[gen])
                logger.info("best params: %s", best.params)

                if best.fitness >= cfg.threshold or gen == cfg.max_generations - 1:
                    break

                self._next_generation(elite=best)
                self._evaluate_population()
                current_best = self._best()
                if current_best.fitness > best.fitness:
                    best = deepcopy(current_best)
        except KeyboardInterrupt:
            logger.warning("GA interrupted at generation %d; "
                           "returning best-so-far result", gen)
            interrupted = True
            if best is None:
                # 初世代の評価中に中断された場合
                evaluated = [c for c in self.population if c.fitness is not None]
                if not evaluated:
                    raise
                best = deepcopy(max(evaluated, key=lambda c: c.fitness))

        return GAResult(
            best_params=dict(best.params),
            best_fitness=float(best.fitness),
            fitness_history=self.fitness_history[:gen + 1].copy(),
            diversity_history=self.diversity_history[:gen + 1].copy(),
            generations_run=gen + 1,
            interrupted=interrupted,
            population=self.population,
        )


def _cpu_count():
    import os
    return os.cpu_count() or 1
