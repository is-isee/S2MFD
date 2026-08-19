"""S2MFD.inference (GAパラメタ推定) のテスト。"""
import numpy as np
import pytest

from S2MFD import inference
from S2MFD.inference.genetic import GAConfig, GeneticAlgorithm
from S2MFD.inference.observations import load_observations, load_initial_field
from S2MFD.inference.problem import (
    DynamoProblem, TimeFunction, linear_plus_sines, U0_SPEC, S0_SPEC)

YEAR_SECONDS = 365 * 24 * 3600


class TestMetrics:
    def test_correlation(self):
        o = np.sin(np.linspace(0, 10, 100))
        assert np.isclose(inference.correlation(o, 2 * o + 1), 1.0)
        assert np.isclose(inference.correlation(o, -o), -1.0)

    def test_nmse_zero_for_identical(self):
        o = np.abs(np.sin(np.linspace(0, 10, 100))) + 0.1
        assert inference.nmse(o, o) == 0.0

    def test_fitness_weights(self):
        o = np.abs(np.sin(np.linspace(0, 10, 100))) + 0.1
        s = o * 1.1
        expected = 0.9 * inference.correlation(o, s) - 0.1 * inference.nmse(o, s)
        assert np.isclose(inference.fitness(o, s), expected)

    def test_period_mse_identical_series(self):
        t = np.arange(200) * 40 * 86400.0
        x = np.sin(2 * np.pi * t / (11 * YEAR_SECONDS))**2 + 0.05
        assert inference.period_mse(x, x, t, t) == 0.0

    def test_period_mse_mismatched_counts(self):
        t = np.arange(200) * 40 * 86400.0
        x = np.sin(2 * np.pi * t / (11 * YEAR_SECONDS))**2 + 0.05
        flat = np.ones_like(x)
        assert inference.period_mse(x, flat, t, t) == np.inf

    def test_mape_guard_against_zero(self):
        o = np.array([0.0, 1.0, 2.0])
        s = np.array([0.1, 1.0, 2.0])
        assert np.isfinite(inference.mape(o, s))


class TestTimeFunction:
    def test_endpoints(self):
        """t=0 で a_start、t=t_span で a_end (sin項は両端でゼロ)。"""
        t_span = 50.0 * YEAR_SECONDS
        assert np.isclose(
            linear_plus_sines(0.0, 700, 800, 30, -20, 10, t_span), 700.0)
        assert np.isclose(
            linear_plus_sines(t_span, 700, 800, 30, -20, 10, t_span), 800.0)

    def test_absolute_time_offset(self):
        t_start = 1000.0 * YEAR_SECONDS
        t_span = 50.0 * YEAR_SECONDS
        f = TimeFunction(700, 800, 30, -20, 10, t_start, t_span)
        assert np.isclose(f(t_start), 700.0)
        assert np.isclose(f(t_start + t_span), 800.0)
        assert np.isclose(
            f(t_start + 0.3 * t_span),
            linear_plus_sines(0.3 * t_span, 700, 800, 30, -20, 10, t_span))


class _Quadratic:
    """GA テスト用の toy 評価器 (picklable)。最大値 0 at (3, -1)。"""

    def __call__(self, params):
        return -((params['x'] - 3.0)**2 + (params['y'] + 1.0)**2)


class TestGeneticAlgorithm:
    SPEC = {'x': (-10.0, 10.0), 'y': (-10.0, 10.0)}

    def _run(self, pop=20, gens=60, seed=0, **cfg_kw):
        config = GAConfig(threshold=-1e-3, max_generations=gens,
                          max_workers=1, seed=seed, **cfg_kw)
        ga = GeneticAlgorithm(_Quadratic(), self.SPEC,
                              population_size=pop, config=config)
        return ga.run()

    def test_converges_on_quadratic(self):
        result = self._run()
        assert result.best_fitness > -0.1
        assert abs(result.best_params['x'] - 3.0) < 0.5
        assert abs(result.best_params['y'] + 1.0) < 0.5

    def test_small_population_does_not_crash(self):
        # 旧実装は個体数<10でトーナメントサイズ0となりクラッシュした
        result = self._run(pop=4, gens=10)
        assert np.isfinite(result.best_fitness)

    def test_reproducible_with_seed(self):
        r1 = self._run(seed=123, gens=15)
        r2 = self._run(seed=123, gens=15)
        assert r1.best_params == r2.best_params
        assert np.array_equal(r1.fitness_history, r2.fitness_history)

    def test_fitness_history_monotone(self):
        # エリート保存により最良適応度は非減少
        result = self._run()
        assert np.all(np.diff(result.fitness_history) >= 0)

    def test_clip_to_bounds(self):
        result = self._run(gens=30, clip_to_bounds=True)
        for chrom in result.population:
            for name, (lo, hi) in self.SPEC.items():
                assert lo <= chrom.params[name] <= hi

    def test_parallel_evaluation(self):
        config = GAConfig(threshold=-1e-3, max_generations=5,
                          max_workers=2, seed=0)
        ga = GeneticAlgorithm(_Quadratic(), self.SPEC,
                              population_size=8, config=config)
        result = ga.run()
        assert np.isfinite(result.best_fitness)


class TestObservations:
    def test_load_obs(self):
        obs = load_observations('OBS')
        assert obs.years[0] == 1700.5
        # 40日間隔
        assert np.allclose(np.diff(obs.years), 40 / 365)
        assert np.allclose(obs.seconds, obs.years * YEAR_SECONDS)

    def test_index_of_year(self):
        obs = load_observations('OBS')
        i = obs.index_of_year(1944)
        assert abs(obs.years[i] - 1944) < 40 / 365

    def test_minima_detection(self):
        obs = load_observations('OBS')
        minima = obs.detect_minima_years()
        assert len(minima) > 20  # ~29 cycles in 324 years
        # 周期はおよそ 8-15 年
        periods = np.diff(minima)
        assert 5 < np.median(periods) < 15

    def test_load_custom_csv(self, tmp_path):
        path = tmp_path / 'my_ssn.csv'
        years = np.arange(1900, 1950, 1.0)
        ssn = 50 + 50 * np.sin(2 * np.pi * years / 11)
        np.savetxt(path, np.column_stack([years, ssn]),
                   delimiter=',', header='Year,SSN')
        obs = load_observations(str(path))
        assert obs.years[0] == 1900
        assert len(obs.ssn) > len(years)  # 内挿で増える

    def test_initial_field_shape(self):
        Bph, Aph = load_initial_field()
        assert Bph.shape == (130, 130)
        assert Aph.shape == (130, 130)


class TestDynamoProblem:
    def test_spec_selection(self):
        obs = load_observations('OBS')
        Bph0, Aph0 = load_initial_field()
        p = DynamoProblem('parameters/inference.py', obs.seconds, obs.ssn,
                          100, 110, 'u0', Bph0, Aph0)
        assert p.param_spec() == U0_SPEC
        with pytest.raises(ValueError, match='stage1'):
            DynamoProblem('parameters/inference.py', obs.seconds, obs.ssn,
                          100, 110, 's0', Bph0, Aph0)

    def test_simulate_smoke(self):
        """短いスピンアップ+短い窓で1個体評価が通ること (128x128 実格子)。"""
        obs = load_observations('OBS')
        Bph0, Aph0 = load_initial_field()
        i0 = obs.index_of_year(1944)
        problem = DynamoProblem(
            'parameters/inference.py', obs.seconds, obs.ssn,
            i0, i0 + 4, 'u0', Bph0, Aph0,
            spinup_years=0.5, until_minimum=False)
        params = {'as_u': 700.0, 'ae_u': 720.0, 'a1_u': 10.0,
                  'a2_u': -5.0, 'a4_u': 2.0, 'a0_s': 50.0}
        sn, ts, sim = problem.simulate(params)
        assert len(sn) == problem.window_length
        assert np.all(np.isfinite(sn))
        assert np.all(np.diff(ts) > 0)
        fitness = problem(params)
        assert np.isfinite(fitness) and -1.1 <= fitness
