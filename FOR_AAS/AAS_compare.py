import matplotlib.pyplot as plt
import numpy as np
import os, sys
sys.path.append('../')
import S2MFD

# 分析範囲、対象は手で決める
datadir = '../OBS_results_no2/data_obs_full/'
n0 = 210
n1 = 2957
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
# 基本ラベルサイズの指定
size = 10

# 日本語使用可能
import matplotlib as mpl
mpl.rcParams['font.family'] = 'IPAPGothic'

# ============================================================================== #
# 黒点数散布図
cc   = np.sum((SN1-np.mean(SN1))*(SN2-np.mean(SN2)))/np.sqrt(np.sum((SN1-np.mean(SN1))**2)*np.sum((SN2-np.mean(SN2))**2))
plt.figure(figsize=(5, 5))
plt.scatter(SN1, SN2, s=4, alpha=0.7)

# 回帰直線（一次関数）
coef = np.polyfit(SN1, SN2, 1)   # 1次多項式 (傾き, 切片)
fit_fn = np.poly1d(coef)
x_line = np.linspace(np.min(SN1), np.max(SN1), 100)
plt.plot(x_line, fit_fn(x_line), color="red", linewidth=2)

# 軸や凡例の整形
plt.ylim(0,250)
plt.xticks([0,50,100,150,200,250],fontsize=1.6*size)  # x軸の数値サイズを調整
plt.yticks(fontsize=1.6*size)  # y軸の数値サイズを調整
plt.xlabel("Observational sunspot numuber", fontsize=1.6*size)
plt.ylabel("Theoretical sunspot numuber", fontsize=1.6*size)
plt.title("(a)", fontsize=2.5*size, loc='left')
plt.grid(True, linestyle="--", alpha=0.6)
plt.axis("equal")
plt.tight_layout()
plt.savefig("correlation_scatter_fit_SN.png", dpi=300)
plt.show()
# ============================================================================== #
# 周期相関関係
def judge3(SN1, SN2, time1, time2):
    """
    Compares the period of sunspots with the simulation results
    """
    from scipy.signal import argrelextrema
    
    # 黒点数の極小を取得
    indices_num = argrelextrema(SN2, np.less, order=30)[0]
    indices_obs = argrelextrema(SN1, np.less, order=30)[0]
    
    # 追加分
    indices_num = np.insert(indices_num, 0, 0) # 先頭に追加
    indices_obs = np.insert(indices_obs, 0, 0) # 先頭に追加
    # indices_num = np.insert(indices_num, len(indices_num),len(Sunspot_N)-1) # 最後に追加
    # indices_obs = np.insert(indices_obs, len(indices_obs),len(self.SN)-1) # 最後に追加
    print("numの極小点のインデックス:", indices_num)
    print("obsの極小点のインデックス:", indices_obs)

    # 時間を調べる
    timet_num = time2[indices_num]
    timet_obs = time1[indices_obs]

    # 周期を計算
    period_num = np.diff(timet_num)/ 365/ 24 / 3600 # 秒から年に変換
    period_obs = np.diff(timet_obs)/ 365/ 24 / 3600 # 秒から年に変換
    print("numの周期:", period_num, "obsの周期:", period_obs)
    
    # 周期の相関係数を計算
    if len(period_num) == len(period_obs):
        period_corr = np.sum((period_num-np.mean(period_num))*(period_obs-np.mean(period_obs)))/np.sqrt(np.sum((period_num-np.mean(period_num))**2)\
                        *np.sum((period_obs-np.mean(period_obs))**2))
        print("周期の相関係数=", period_corr)
    else:
        print("周期の長さが異なるため、相関係数を計算できません。")
        period_corr = 0.0
    
    return period_corr, period_num, period_obs, indices_obs, indices_num

period_corr, period_num, period_obs, obs_min, num_min = judge3(SN1, SN2, time1*365*24*3600, time2*365*24*3600)
plt.figure(figsize=(5, 5))
plt.scatter(period_num, period_obs, s=4, alpha=0.7)

# 回帰直線（一次関数）
coef = np.polyfit(period_num, period_obs, 1)   # 1次多項式 (傾き, 切片)
fit_fn = np.poly1d(coef)

# 線を描画（x軸の範囲に合わせる）
x_line = np.linspace(np.min(period_num), np.max(period_num), 100)
plt.plot(x_line, fit_fn(x_line), color="red", linewidth=2)
plt.title("(b)", fontsize=2.5*size, loc='left')
# 軸や凡例の整形
plt.xlabel(r"Observational period$~[\mathrm{yr}]$", fontsize=16)
plt.ylabel(r"Theoretical period$~[\mathrm{yr}]$", fontsize=16)
plt.xticks(fontsize=1.6*size)  # x軸の数値サイズを調整
plt.yticks(fontsize=1.6*size)  # y軸の数値サイズを調整
plt.grid(True, linestyle="--", alpha=0.6)
# plt.axis("equal")
plt.tight_layout()
plt.savefig("correlation_scatter_fit_P.png", dpi=300)
plt.show()
# ============================================================================== #
# 極大値相関関係
# 周期相関関係
def maximum(SN1, SN2):
    """
    Compares the period of sunspots with the simulation results
    """
    from scipy.signal import argrelextrema
    
    # 黒点数の極小を取得
    indices_num = argrelextrema(SN2, np.greater, order=30)[0]
    indices_obs = argrelextrema(SN1, np.greater, order=30)[0]
    # TODO 一番最後の要素を削除（2024年を極大期と見做さない）便宜調節
    # indices_num = indices_num[:-1]
    SN1_max = SN1[indices_obs]
    SN2_max = SN2[indices_num]

    SN_corr = 0.0
    print(SN1_max, SN2_max)
    SN_corr = np.sum((SN1_max-np.mean(SN1_max))*(SN2_max-np.mean(SN2_max)))/np.sqrt(np.sum((SN1_max-np.mean(SN1_max))**2)\
                        *np.sum((SN2_max-np.mean(SN2_max))**2))
    print("極大値の相関係数=", SN_corr)
    return SN1_max, SN2_max, SN_corr, indices_obs, indices_num
SN1_max, SN2_max, SN_corr, obs_max, num_max = maximum(SN1, SN2)

plt.figure(figsize=(5, 5))
plt.scatter(SN1_max, SN2_max, s=4, alpha=0.7)

# 回帰直線（一次関数）
coef = np.polyfit(SN1_max, SN2_max, 1)   # 1次多項式 (傾き, 切片)
fit_fn = np.poly1d(coef)

# 線を描画（x軸の範囲に合わせる）
x_line = np.linspace(np.min(SN1_max), np.max(SN1_max), 100)
plt.plot(x_line, fit_fn(x_line), color="red", linewidth=2)

# 軸や凡例の整形
plt.title("(c)", fontsize=2.5*size, loc='left')
plt.xlabel("Observational sunspot number", fontsize=16)
plt.ylabel("Theoretical sunspot number", fontsize=16)
plt.xticks(fontsize=1.6*size)  # x軸の数値サイズを調整
plt.yticks(fontsize=1.6*size)  # y軸の数値サイズを調整
# plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("correlation_scatter_fit_maximum.png", dpi=300)
plt.show()

# ============================================================================== #
# 黒点数の比較グラフ
# グラフの描画
plt.figure(figsize=(32, 8))  # グラフのサイズを調整
plt.plot(time1, SN1, 'r', label='observation',linewidth=2.5)
plt.plot(time2, SN2, 'b', label='GA inference',linewidth=2.5)
# 極小点
plt.plot(time1[obs_min], SN1[obs_min], 'o', color='orange', label='obs. min', markersize=10)
plt.plot(time2[num_min], SN2[num_min], 'o', color='magenta', label='GA. min', markersize=10)

# 極大点
plt.plot(time1[obs_max], SN1[obs_max], '^', color='lime', label='obs. max', markersize=10)
plt.plot(time2[num_max], SN2[num_max], '^', color='k', label='GA. max', markersize=10)
print(len(obs_max), len(num_max))

# 軸ラベル・タイトル
# plt.title('黒点数時間変化', fontsize=4*size)  # タイトルを追加
plt.xlabel(r'$t~[\mathrm{yr}]$', fontsize=5*size)
plt.ylabel('Sunspot number', fontsize=4*size)

# 軸のメモリフォントサイズ
plt.xticks(fontsize=4*size)  # x軸の数値サイズを調整
plt.yticks(fontsize=4*size)  # y軸の数値サイズを調整
plt.title("(d)", fontsize=5*size, loc='left')
# グリッドを追加して見やすく
plt.grid(True, linestyle='--', alpha=0.7)
# 凡例を表示
plt.legend(fontsize=2*size, loc='upper right')  # 凡例を右上に固定
plt.tight_layout()
# グラフを保存
plt.savefig("P_sunspots_number_compare.png", dpi=300)  # 解像度を高める
plt.clf()

# ============================================================================== #
# u0の比較グラフ
# plt.plot(time2,uu0t2,'o',color='b',label='estimation',linestyle='None',markersize='2')
plt.plot(time2, uu0t2, 'b', label='GA inference',linewidth=2.5)
plt.xlabel('Time(years)',fontsize=3*size)
plt.ylabel(r'$u_0~[\rm{cm~s^{-1}}]$',fontsize=3*size)
# plt.title(r'$u_0$ 時間変化', fontsize=4*size)  # タイトルを追加
ymax = max(np.max(uu0t2), np.max(uu0t2)) * 1.1
plt.ylim(bottom=0, top=ymax)
# 軸のメモリを細かく設定
plt.xticks(fontsize=2*size)  # x軸の数値サイズを調整
plt.yticks(fontsize=2*size)  # y軸の数値サイズを調整
# グリッドを追加して見やすく
plt.grid(True, linestyle='--', alpha=0.7)
# 凡例を表示
plt.legend(fontsize=1.6*size, loc='upper right')  # 凡例を右上に固定
# グラフを保存
plt.savefig("P_u0_compare.png", dpi=300)
plt.clf()

# ============================================================================== #
# so0の比較グラフ
# plt.plot(time2,so0t2,'o',color='b',label='estimation',linestyle='None',markersize='2')
plt.plot(time2, so0t2, 'b', label='GA inference',linewidth=2.5)
ymax = max(np.max(so0t2), np.max(so0t2)) * 1.1
plt.ylim(bottom=0, top=ymax)
plt.xlabel('Time(years)',fontsize=3*size)
plt.ylabel(r'$s_0(\rm{cm/s})$',fontsize=3*size)
# plt.title(r'$s_0$ 時間変化', fontsize=4*size)  # タイトルを追加
# 軸のメモリを細かく設定
plt.xticks(fontsize=2*size)  # x軸の数値サイズを調整
plt.yticks(fontsize=2*size)  # y軸の数値サイズを調整
# グリッドを追加して見やすく
plt.grid(True, linestyle='--', alpha=0.7)
# 凡例を表示
plt.legend(fontsize=1.6*size, loc='upper right')  # 凡例を右上に固定

# 極小点をマーク（後で消す）
plt.plot(time2[[180,763,1598]], so0t2[[180,763,1598]], 'o', color='orange', label='obs. min', markersize=10)
# グラフを保存
plt.savefig("P_s0_compare.png", dpi=300)
plt.clf()

# ============================================================================== #
# SN,u0,s0のマルチパネル
size2 = 4
import matplotlib.pyplot as plt

# fig, axes = plt.subplots(3, 1, figsize=(24, 16), sharex=True)  # 3行1列, x軸共有
fig, axes = plt.subplots(3, 1, figsize=(24, 16))  # 3行1列, x軸共有しない

# --- 一番上 ---
axes[0].plot(time1, SN1, 'r', label='observation', linewidth=2.5)
axes[0].plot(time2, SN2, 'b', label='GA inference', linewidth=2.5)
# axes[0].set_xlabel(r"$t~[\rm{yr}]$", fontsize=15*size2)
axes[0].set_ylabel("Sunspot number", fontsize=8*size2)
axes[0].legend(loc="upper right", fontsize=6*size2)
axes[0].tick_params(axis='both', labelsize=8*size2)
axes[0].grid(True, linestyle="--", alpha=0.6)
axes[0].set_title("(a)", fontsize=10*size2, loc='left')
# --- 中央 ---
axes[1].plot(time2, uu0t2, 'b', linewidth=2.5)
ymax = max(np.max(uu0t2), np.max(uu0t2)) * 1.1
axes[1].set_ylim(bottom=0, top=ymax)
# axes[1].set_xlabel(r"$t~[\rm{yr}]$", fontsize=15*size2)
axes[1].set_ylabel(r'$u_0~[\rm{cm~s^{-1}}]$', fontsize=12*size2)
# axes[1].legend(loc="upper right", fontsize=10*size2)
axes[1].tick_params(axis='both', labelsize=8*size2)
axes[1].grid(True, linestyle="--", alpha=0.6)
axes[1].set_title("(b)", fontsize=10*size2, loc='left')
# --- 一番下 ---
axes[2].plot(time2, so0t2, 'b', linewidth=2.5)
ymax = max(np.max(so0t2), np.max(so0t2)) * 1.1
axes[2].set_ylim(bottom=0, top=ymax)
axes[2].set_xlabel(r"$t~[\rm{yr}]$", fontsize=12*size2)
axes[2].set_ylabel(r'$s_0~[\rm{cm~s^{-1}}]$', fontsize=12*size2)
# axes[2].set_xlabel("Years", fontsize=12*size2)
# axes[2].legend(loc="upper right", fontsize=10*size2)
axes[2].tick_params(axis='both', labelsize=8*size2)
axes[2].grid(True, linestyle="--", alpha=0.6)
axes[2].set_title("(c)", fontsize=10*size2, loc='left')
# レイアウト調整
plt.tight_layout()
plt.savefig("multi_panel.png", dpi=300)
plt.show()
# ============================================================================== #

sd   = np.sqrt((np.sum(SN1) - np.sum(SN2))**2) / np.sum(SN1)
sd2  = np.sum(np.sqrt((SN1 - SN2)**2)) / np.sum(SN1)
MAPE = np.sum(abs((SN1 - SN2) / SN1)) / len(SN1)
NMSE = np.sum((SN1 - SN2)**2) / np.sum(SN1**2)
print("----------------------------------------------")
print("相関係数＝",cc,"総数誤差割合＝",sd,"(誤差割合はGA期間中の和の誤差)","黒点誤差(自身で考案)=",sd2,"MAPE=",MAPE,"NMSE=",NMSE)
print("評価関数=",alpha*cc-(1-alpha)*NMSE)
