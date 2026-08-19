"""
S2MFD.inference: 遺伝的アルゴリズム (GA) によるダイナモモデルの
パラメタ推定パッケージ。

Shimizu & Hotta (2026), ApJ 996, 102 の手法の実装。
子午面流振幅 u0(t) と Babcock-Leighton α効果振幅 s0(t) の時間変化を、
黒点数観測 (SILSO) との比較で2段階推定する。

Example
-------
>>> from S2MFD import inference
>>> obs = inference.load_observations('OBS')
>>> # コマンドラインからは: python -m S2MFD.inference.cli --help
"""
from .observations import Observations, load_observations, detect_minimum_years
from .metrics import correlation, nmse, total_count_error, period_mse, mape, fitness
from .genetic import GAConfig, GeneticAlgorithm, GAResult
from .problem import DynamoProblem, linear_plus_sines, U0_SPEC, S0_SPEC
from .compare import compare_with_observation

__all__ = [
    'Observations', 'load_observations', 'detect_minimum_years',
    'correlation', 'nmse', 'total_count_error', 'period_mse', 'mape', 'fitness',
    'GAConfig', 'GeneticAlgorithm', 'GAResult',
    'DynamoProblem', 'linear_plus_sines', 'U0_SPEC', 'S0_SPEC',
    'compare_with_observation',
]
