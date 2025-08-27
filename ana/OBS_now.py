import matplotlib.pyplot as plt
import numpy as np
import os, sys
sys.path.append('../')
import S2MFD

# 分析範囲、対象は手で決める
datadir = '../OBS_results/data_obs_6/'
datadir1 = '../data_obs_random/'
n0 = 210
n1 = 686
alpha = 0.9  # 評価関数の重み

data = S2MFD.Data.initial_load(datadir)

cfg = data.cfg
grid = data.grid
setup = data.setup

# ============================================================================== #
# 黒点などを割り出すための関数
base  = 1+np.argmin(abs(grid.rr-0.7*cfg.RSUN))
# 移動平均関数
def moving_average(SunspotsNum, n_conv):
    conv_f = np.ones(n_conv)/n_conv
    SN_smooth = np.convolve(SunspotsNum, conv_f, mode='same')#移動平均
    
    return SN_smooth
# 磁場エネルギー密度B^2(toroidal,15°,r_c)を黒点数の指標とする。
def Karak_deffine(Bpht,base,cfg,grid):
    gamma = 5.8653520852  # 観測に基づいた調整パラメタ
    SN = gamma*Bpht[base,np.argmin(abs(grid.th-75/180*np.pi)),:]**2
    SN2 = moving_average(SN, 4)
    return SN

# 磁場エネルギー密度B^2(toroidal,10°~20°,r_c)を黒点数の指標とする。
def original_deffine(Bpht,base,cfg,grid):
    locap =   np.argmin(abs(grid.th- 80/180*np.pi))
    locam =   np.argmin(abs(grid.th- 70/180*np.pi))
    SN = np.mean(Bpht[base,locam:locap,:]**2,axis=0)
    SN2 = moving_average(SN, 4)
    return SN2
# ============================================================================== #
# 観測データの読み込み
data = np.genfromtxt("../obs_data/obs_data/SN_Yearly_interp.csv", delimiter=',', skip_header=1)
time1 = data[n0:n1, 1]/24/60/60/365
SN1 = data[n0:n1, 3]
# ============================================================================== #
# datadir2
data = S2MFD.Data.initial_load(datadir)

cfg = data.cfg
grid = data.grid
setup = data.setup

fig = plt.figure('dynamo',figsize=(10,10))
tau_diff = data.cfg.RSUN**2/data.cfg.ett
timet = np.zeros(n1-n0)
nt = np.zeros(n1-n0)
ndt = np.zeros(n1-n0)
Brrt = np.zeros((grid.ixg,grid.jxg,n1-n0))
Btht = np.zeros((grid.ixg,grid.jxg,n1-n0))
Bpht = np.zeros((grid.ixg,grid.jxg,n1-n0))
Apht = np.zeros((grid.ixg,grid.jxg,n1-n0))
so0t = np.zeros(n1-n0)
uu0t = np.zeros(n1-n0)
dltt = np.zeros(n1-n0)

for n  in range(n0,n1):
    print(n)
    data.data_load(n)
    Brr, Bth = S2MFD.physics.poloidal_mag(data.Aph, grid.RR, grid.sinTH, grid.drr, grid.dth)
    d = np.load(file=datadir+'data.'+str(n).zfill(6)+'.npz')
    timet[n-n0] = d['time']
    nt[n-n0] = d['n']
    ndt[n-n0] = d['nd']
    Brrt[:,:,n-n0] = Brr
    Btht[:,:,n-n0] = Bth
    Bpht[:,:,n-n0] = d['Bph']
    Apht[:,:,n-n0] = d['Aph']
    so0t[n-n0] = d['so0']
    uu0t[n-n0] = d['uu0']
    if 'dl' in d:
        dltt[n-n0] = d['dl']
        
SN2 = Karak_deffine(Bpht,base,cfg,grid)
so0t2= so0t
uu0t2 = uu0t
time2 = timet/data.cfg.d2s/365
# ============================================================================== #
# ============================================================================== #
# datadir2
data = S2MFD.Data.initial_load(datadir1)

cfg = data.cfg
grid = data.grid
setup = data.setup

fig = plt.figure('dynamo',figsize=(10,10))
tau_diff = data.cfg.RSUN**2/data.cfg.ett
timet = np.zeros(n1-n0)
nt = np.zeros(n1-n0)
ndt = np.zeros(n1-n0)
Brrt = np.zeros((grid.ixg,grid.jxg,n1-n0))
Btht = np.zeros((grid.ixg,grid.jxg,n1-n0))
Bpht = np.zeros((grid.ixg,grid.jxg,n1-n0))
Apht = np.zeros((grid.ixg,grid.jxg,n1-n0))
so0t = np.zeros(n1-n0)
uu0t = np.zeros(n1-n0)
dltt = np.zeros(n1-n0)

for n  in range(n0,n1):
    print(n)
    data.data_load(n)
    Brr, Bth = S2MFD.physics.poloidal_mag(data.Aph, grid.RR, grid.sinTH, grid.drr, grid.dth)
    d = np.load(file=datadir1+'data.'+str(n).zfill(6)+'.npz')
    timet[n-n0] = d['time']
    nt[n-n0] = d['n']
    ndt[n-n0] = d['nd']
    Brrt[:,:,n-n0] = Brr
    Btht[:,:,n-n0] = Bth
    Bpht[:,:,n-n0] = d['Bph']
    Apht[:,:,n-n0] = d['Aph']
    so0t[n-n0] = d['so0']
    uu0t[n-n0] = d['uu0']
    if 'dl' in d:
        dltt[n-n0] = d['dl']
        
SN3 = Karak_deffine(Bpht,base,cfg,grid)
so0t3= so0t
uu0t3 = uu0t
time3 = timet/data.cfg.d2s/365

# 基本ラベルサイズの指定
size = 10

# 日本語使用可能
import matplotlib as mpl
mpl.rcParams['font.family'] = 'IPAPGothic'
# ============================================================================== #
# 黒点数の比較グラフ
# グラフの描画
plt.figure(figsize=(10, 6))  # グラフのサイズを調整
plt.plot(time1, SN1, 'salmon',lw=6,label='観測')
plt.plot(time3, SN3, 'b--',label='推定初期')
plt.plot(time2, SN2, 'b',lw=2.5,label='最適解')
# 軸ラベル・タイトル
plt.title('黒点数時間変化', fontsize=4*size)  # タイトルを追加
plt.xlabel('年', fontsize=3*size)
plt.ylabel('黒点相対数', fontsize=3*size)

# 軸のメモリフォントサイズ
plt.xticks(fontsize=2*size)  # x軸の数値サイズを調整
plt.yticks(fontsize=2*size)  # y軸の数値サイズを調整

# グリッドを追加して見やすく
plt.grid(True, linestyle='--', alpha=0.7)
# 凡例を表示
plt.legend(fontsize=1.6*size, loc='upper right')  # 凡例を右上に固定
plt.tight_layout()
# グラフを保存
plt.savefig("P_sunspots_number_compare_forpre.png", dpi=300)  # 解像度を高める
plt.clf()
