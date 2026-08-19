"""観測とシミュレーションの黒点数時系列を比較する評価指標 (純関数)。

旧 IDPA の Simulation.judge1〜judge5 を副作用なし (プロット・print なし) の
関数として移植したもの。
"""
import numpy as np
from scipy.signal import argrelextrema

YEAR_SECONDS = 365 * 24 * 3600


def correlation(obs, sim):
    """ピアソン相関係数 (旧 judge)。"""
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    do = obs - np.mean(obs)
    ds = sim - np.mean(sim)
    denom = np.sqrt(np.sum(do**2) * np.sum(ds**2))
    if denom == 0:
        return 0.0
    return float(np.sum(do * ds) / denom)


def total_count_error(obs, sim):
    """黒点総数の相対誤差 (旧 judge2)。"""
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    return float(abs(np.sum(obs) - np.sum(sim)) / np.sum(obs))


def period_mse(obs, sim, t_obs, t_sim, order=30):
    """周期の平均二乗誤差 [年^2] (旧 judge3 の中身、符号は正)。

    観測・シミュレーションそれぞれの黒点数極小の間隔から周期を求め、
    その差の二乗平均を返す。周期の数が一致しない場合は np.inf を返す。

    Notes
    -----
    旧実装は観測の極小インデックスにシミュレーションの時刻を(およびその逆を)
    対応させるバグがあった (コード内 TODO で自認)。本実装では
    観測の極小は t_obs、シミュレーションの極小は t_sim と正しく対応させる。
    """
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    t_obs = np.asarray(t_obs, dtype=float)
    t_sim = np.asarray(t_sim, dtype=float)

    idx_sim = argrelextrema(sim, np.less, order=order)[0]
    idx_obs = argrelextrema(obs, np.less, order=order)[0]

    # 両端を極小として追加 (推定区間は極小期から極小期までのため)
    idx_sim = np.concatenate(([0], idx_sim, [len(sim) - 1]))
    idx_obs = np.concatenate(([0], idx_obs, [len(obs) - 1]))

    period_sim = np.diff(t_sim[idx_sim]) / YEAR_SECONDS
    period_obs = np.diff(t_obs[idx_obs]) / YEAR_SECONDS

    if len(period_sim) != len(period_obs):
        return np.inf
    return float(np.mean((period_sim - period_obs)**2))


def nmse(obs, sim):
    """正規化平均二乗誤差 (旧 judge4)。"""
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    return float(np.sum((obs - sim)**2) / np.sum(obs**2))


def mape(obs, sim, eps=1e-10):
    """平均絶対パーセント誤差 (旧 judge5)。

    Notes
    -----
    旧実装は観測値がゼロ近傍 (極小期) で発散した。eps でガードするが、
    黒点数のように 0 を含む時系列には本質的に不向きな指標である。
    """
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    return float(np.mean(np.abs((obs - sim) / np.maximum(np.abs(obs), eps))))


def fitness(obs, sim, alpha=0.9):
    """適応度 F = alpha * 相関係数 - (1 - alpha) * NMSE (論文式 3.1)。"""
    return alpha * correlation(obs, sim) - (1 - alpha) * nmse(obs, sim)
