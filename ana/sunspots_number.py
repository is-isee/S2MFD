"""黒点数プロキシの時系列を描くスクリプト。

使い方:
    python sunspots_number.py [datadir]   # 既定は ../data/
"""
import matplotlib.pyplot as plt
import numpy as np

from ana_common import get_datadir, load_run, radial_index

datadir = get_datadir()
run = load_run(datadir, with_poloidal=False)
cfg, grid, timet, Bpht = run.cfg, run.grid, run.timet, run.Bpht
n1 = run.n1

# 黒点数の計上
# TODO 範囲はよく吟味
N_num = np.zeros(n1)
S_num = np.zeros(n1)
thrsh = 3.5
# ゴーストを踏まない添字 (2026-08-23)。以前の 1+argmin(...) は
# 1 セル外側を指していた。
base  = radial_index(grid, 0.7*cfg.RSUN)
# グリッドに応じた北半球と南半球の分割
if grid.jxg % 2==0:
    S_equa = grid.jxg//2 - 1
    N_equa = S_equa + 1
else:
    S_equa = (grid.jxg-1)//2 - 1
    N_equa = S_equa + 2

# n1は時間要素の最後の番号
for t in range(n1):
    S_num[t] = np.sum(Bpht[base,grid.margin:S_equa+1,t]<-thrsh) + np.sum(Bpht[base,grid.margin:S_equa+1,t]>thrsh)
    N_num[t] = np.sum(Bpht[base,N_equa:grid.jxg-grid.margin,t]<-thrsh) + np.sum(Bpht[base,N_equa:grid.jxg-grid.margin,t]>thrsh)

# 観測に基づいた調整パラメタ
kappa = 0.3
S_num = kappa * S_num
N_num = kappa * N_num

n_conv = 4 #移動平均の個数
conv_f = np.ones(n_conv)/n_conv

S_num2 = np.convolve(S_num, conv_f, mode='same')#移動平均
N_num2 = np.convolve(N_num, conv_f, mode='same')#移動平均

# 実際のスナップショット時刻を使う (旧実装は tend からの等間隔を仮定していた)
time_y = timet / cfg.d2s / 365
plt.plot(time_y, S_num2, 'r', label='sunspots number')
plt.xlabel('time(year)')
plt.ylabel('sunspots number')
plt.legend()
plt.show()
