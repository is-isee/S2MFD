import numpy as np

# .npz ファイルを開く
datadir = "../data/"
d = np.load(datadir + "data.000761.npz")

data = S2MFD.Data.initial_load(datadir)

cfg = data.cfg
grid = data.grid
setup = data.setup

# 各データを展開
Bph = d["Bph"]
Aph = d["Aph"]
time = d["time"]
n = d["n"]


base  = 1+np.argmin(abs(data.grid.rr-0.8*data.cfg.RSUN))
# グリッドに応じた北半球と南半球の分割
if grid.jxg % 2==0: 
    S_equa = data.grid.jxg//2 - 1
    N_equa = S_equa + 1
else:
    S_equa = (data.grid.jxg-1)//2 - 1
    N_equa = S_equa + 2

gamma_s = 0.4
gamma_n = 0.0
Aph[base:,:S_equa] = gamma_s * Aph[base:,:S_equa]
Aph[base:,N_equa:] = gamma_n * Aph[base:,N_equa:]

# 修正後のデータを新しい .npz に保存
np.savez(datadir + "data.000761.npz", Bph=Bph, Aph=Aph, time=time, n=n)
