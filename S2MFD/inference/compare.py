"""推定結果と観測の比較プロット・数値出力 (旧 OBS_compare の関数化)。

旧実装との差異:
- 観測データ・出力先を引数で受ける (CWD 依存のハードコードパスを排除)
- matplotlib は Agg 固定 (ヘッドレス・並列環境で安全)
"""
import logging
import os
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from . import metrics
from .problem import linear_plus_sines

logger = logging.getLogger(__name__)

YEAR_SECONDS = 365 * 24 * 3600


def compare_with_observation(obs, index_start, index_end,
                             sn_sim, ts_sim, uu0_sim, so0_sim,
                             params, mode, output_dir, alpha=0.9):
    """推定結果の比較プロットと数値結果を output_dir に保存する。

    Parameters
    ----------
    obs : observations.Observations
        観測時系列
    index_start, index_end : int
        推定窓のインデックス範囲
    sn_sim, ts_sim : numpy.ndarray
        シミュレーションの黒点数プロキシ時系列と時刻 [s]
    uu0_sim, so0_sim : numpy.ndarray
        各出力時点での u0, s0 振幅
    params : dict
        推定されたゲノム
    mode : str
        'u0' or 's0'
    output_dir : str
        保存先ディレクトリ
    alpha : float
        適応度の重み

    Returns
    -------
    dict
        {'cc': 相関係数, 'nmse': NMSE, 'eva': 適応度}
    """
    os.makedirs(output_dir, exist_ok=True)
    time_obs_yr = obs.seconds[index_start:index_end + 1] / YEAR_SECONDS
    ssn_obs = obs.ssn[index_start:index_end + 1]
    time_sim_yr = np.asarray(ts_sim) / YEAR_SECONDS

    size = 10

    # 黒点数の比較
    plt.figure(figsize=(10, 6))
    plt.plot(time_obs_yr, ssn_obs, 'r', label='observation', linewidth=2.5)
    plt.plot(time_sim_yr, sn_sim, 'b', label='GA inference', linewidth=2.5)
    plt.xlabel(r'$t~[\rm{yr}]$', fontsize=3 * size)
    plt.ylabel(r'$\rm{SSN}$', fontsize=3 * size)
    plt.xticks(fontsize=2 * size)
    plt.yticks(fontsize=2 * size)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=1.6 * size)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'P_sunspots_number_compare.png'), dpi=300)
    plt.close()

    # u0 の時間変化
    plt.figure(figsize=(10, 6))
    plt.plot(time_sim_yr, uu0_sim, 'b', label='GA inference')
    plt.xlabel(r'$t~[\rm{yr}]$', fontsize=3 * size)
    plt.ylabel(r'$u_0~[\rm{cm/s}]$', fontsize=3 * size)
    plt.ylim(bottom=0, top=np.max(uu0_sim) * 1.1)
    plt.xticks(fontsize=2 * size)
    plt.yticks(fontsize=2 * size)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'P_u0_compare.png'), dpi=300)
    plt.close()

    # s0 の時間変化
    plt.figure(figsize=(10, 6))
    plt.plot(time_sim_yr, so0_sim, 'b', label='GA inference')
    plt.xlabel(r'$t~[\rm{yr}]$', fontsize=3 * size)
    plt.ylabel(r'$s_0~[\rm{cm/s}]$', fontsize=3 * size)
    plt.ylim(bottom=0, top=np.max(so0_sim) * 1.1)
    plt.xticks(fontsize=2 * size)
    plt.yticks(fontsize=2 * size)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'P_s0_compare.png'), dpi=300)
    plt.close()

    # 推定された時間変化関数の成分分解
    _plot_split_functions(params, mode, ts_sim, output_dir, size)

    # 数値結果
    cc = metrics.correlation(ssn_obs, sn_sim)
    sd = metrics.nmse(ssn_obs, sn_sim)
    eva = alpha * cc - (1 - alpha) * sd
    out_path = os.path.join(output_dir, 'numerical_result.txt')
    with open(out_path, 'a', encoding='utf-8') as fout:
        ts = datetime.now().isoformat(sep=' ', timespec='seconds')
        fout.write(f"{ts}\tcc={cc:.6f}\tNMSE={sd:.6f}\tEVA={eva:.6f}\n")
    logger.info("comparison saved to %s (cc=%.4f NMSE=%.4f EVA=%.4f)",
                output_dir, cc, sd, eva)
    return {'cc': cc, 'nmse': sd, 'eva': eva}


def _plot_split_functions(params, mode, ts_sim, output_dir, size):
    """推定関数を線形成分と各 sin 成分に分解して描画する。"""
    keys = ('as_u', 'ae_u', 'a1_u', 'a2_u', 'a4_u') if mode == 'u0' \
        else ('as_s', 'ae_s', 'a1_s', 'a2_s', 'a4_s')
    if not all(k in params for k in keys):
        return
    a_s, a_e, a1, a2, a4 = (float(params[k]) for k in keys)
    t = np.asarray(ts_sim) - ts_sim[0]
    t_span = float(t[-1])
    if t_span <= 0:
        return
    omega = 2 * np.pi / (t_span * 2)
    lin = (a_s * (t_span - t) + a_e * t) / t_span
    sin1 = a1 * np.sin(1.0 * omega * t)
    sin2 = a2 * np.sin(2.0 * omega * t)
    sin4 = a4 * np.sin(4.0 * omega * t)

    t_yr = t / YEAR_SECONDS
    plt.figure(figsize=(10, 6))
    plt.plot(t_yr, lin + sin1 + sin2 + sin4, 'g', linewidth=3, label='total')
    plt.plot(t_yr, lin + sin1, 'r--', label='l=1')
    plt.plot(t_yr, lin + sin2, 'm--', label='l=2')
    plt.plot(t_yr, lin + sin4, 'c--', label='l=4')
    plt.plot(t_yr, lin, 'k--', label='linear')
    plt.xlabel(r'$t~[\rm{yr}]$', fontsize=2 * size)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=1.6 * size, loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'split_functions_now.png'), dpi=300)
    plt.close()


def plot_history(result, output_dir):
    """GA の適応度・多様性の履歴を保存する。"""
    os.makedirs(output_dir, exist_ok=True)
    for values, name, label in (
            (result.fitness_history, 'fitness_time.png', 'Fitness'),
            (result.diversity_history, 'diversity_time.png', 'Diversity')):
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(len(values)), values, 'r', label=label)
        plt.xlabel('generation number', fontsize=20)
        plt.ylabel(label.lower(), fontsize=20)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=14, loc='upper right')
        plt.savefig(os.path.join(output_dir, name), dpi=300)
        plt.close()
