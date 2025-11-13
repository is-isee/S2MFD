import matplotlib.pyplot as plt
import numpy as np
import os, sys
sys.path.append('../')
import S2MFD

# 分析範囲、対象は手で決める
datadir = '../OBS_results_no4/data_yr_1775_1833/'
n0 = 685
n1 = 1215
inf_year = 52.05479452054692
alpha = 0.9  # 評価関数の重み
################################################
# split_function_prot
# parameters.txtからパラメータを自動読み取り
# param_path = datadir+'parameters.txt'
# params = np.loadtxt(param_path)
# # u0読取の場合
# if len(params) == 5:
#     a_0, a_1, a_2, a_4 = params[1:]
# # so0読取の場合
# else:
#     a_0, a_1, a_2, a_4 = params
# inf_year = 52
################################################
# split_function_now
# parameters.txtからパラメータを自動読み取り
param_path = datadir+'parameters.txt'
params = np.loadtxt(param_path)
# u0読取の場合
if len(params) == 6:
    a_s, a_e, a_1, a_2, a_4 = params[0:5]
# so0読取の場合
else:
    a_s, a_e, a_1, a_2, a_4 = params

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
    # d = np.load(file=datadir+'data.'+str(n).zfill(6)+'.npz')
    d = np.load(file=datadir+'data.'+str(n).zfill(6)+'.npz', allow_pickle=True)
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
# mpl.rcParams['font.family'] = 'IPAPGothic'
# ============================================================================== #
# 黒点数の比較グラフ
# グラフの描画
plt.figure(figsize=(10, 6))  # グラフのサイズを調整
plt.plot(time1, SN1, 'r', label='observation',linewidth=2.5)
plt.plot(time2, SN2, 'b', label='GA inference',linewidth=2.5)

# 軸ラベル・タイトル
# plt.title('黒点数時間変化', fontsize=4*size)  # タイトルを追加
plt.xlabel(r'$t~[\rm{yr}]$', fontsize=3*size)
plt.ylabel(r'$\rm{SSN}$', fontsize=3*size)

# 軸のメモリフォントサイズ
plt.xticks(fontsize=2*size)  # x軸の数値サイズを調整
plt.yticks(fontsize=2*size)  # y軸の数値サイズを調整

# グリッドを追加して見やすく
plt.grid(True, linestyle='--', alpha=0.7)
# 凡例を表示
# plt.legend(fontsize=1.6*size, loc='upper right')  # 凡例を右上に固定
plt.tight_layout()
# グラフを保存
plt.savefig("P_sunspots_number_compare.png", dpi=300)  # 解像度を高める
plt.clf()

# ============================================================================== #
# u0の比較グラフ
plt.plot(time2,uu0t2,'b',label='GA inference')
plt.xlabel(r'$t~[\rm{yr}]$', fontsize=3*size)
plt.ylabel(r'$u_0(\rm{cm/s})$',fontsize=3*size)
# plt.title(r'$u_0$ 時間変化', fontsize=4*size)  # タイトルを追加
ymax = max(np.max(uu0t2), np.max(uu0t2)) * 1.1
plt.ylim(bottom=0, top=ymax)
# 軸のメモリを細かく設定
plt.xticks(fontsize=2*size)  # x軸の数値サイズを調整
plt.yticks(fontsize=2*size)  # y軸の数値サイズを調整
# グリッドを追加して見やすく
plt.grid(True, linestyle='--', alpha=0.7)
# 凡例を表示
# plt.legend(fontsize=1.6*size, loc='upper right')  # 凡例を右上に固定
# グラフを保存
plt.savefig("P_u0_compare.png", dpi=300)
plt.clf()

# ============================================================================== #
# so0の比較グラフ
plt.plot(time2,so0t2,'b',label='GA inference')
ymax = max(np.max(so0t2), np.max(so0t2)) * 1.1
plt.ylim(bottom=0, top=ymax)
# plt.xlabel('年',fontsize=3*size)
plt.xlabel(r'$t~[\rm{yr}]$', fontsize=3*size)
plt.ylabel(r'$s_0(\rm{cm/s})$',fontsize=3*size)
# plt.title(r'$s_0$ 時間変化', fontsize=4*size)  # タイトルを追加
# 軸のメモリを細かく設定
plt.xticks(fontsize=2*size)  # x軸の数値サイズを調整
plt.yticks(fontsize=2*size)  # y軸の数値サイズを調整
# グリッドを追加して見やすく
plt.grid(True, linestyle='--', alpha=0.7)
# 凡例を表示
# plt.legend(fontsize=1.6*size, loc='upper right')  # 凡例を右上に固定
# グラフを保存
plt.tight_layout()
plt.savefig("P_s0_compare.png", dpi=300)
plt.clf()


# =============================================================================== #
# split_function_prot 100にして間違えたもの
def split_function_prot(a_0,a_1,a_2,a_3,inf_year,time2):
    # time２が年単位なのでd2s(day to second)と365をかけて元の単位に戻す
    time = (time2 - time2[0])*data.cfg.d2s*365
    omega = 2*np.pi/(inf_year*365*60*60*100)
    sin1 = a_1*np.sin(1.0*omega*time)
    sin2 = a_2*np.sin(2.0*omega*time)
    sin3 = a_3*np.sin(3.0*omega*time)
    plt.plot(time,a_0+sin1+sin2+sin3,'g',linewidth=3,label='元関数')
    plt.plot(time,a_0+sin1,'r--',label='1次成分')
    plt.plot(time,a_0+sin2,'m--',label='2次成分')
    plt.plot(time,a_0+sin3,'c--',label='3次成分')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=1.6*size, loc='upper right')  # 凡例を右上に固定
    plt.tight_layout()
    plt.savefig("split_functions_prot.png", dpi=300)  # 解像度を高める
    plt.clf()
# split_function_prot(a_0,a_1,a_2,a_3,inf_year,time2)
# ============================================================================== #
# split_function_now
mpl.rcParams['font.family'] = 'IPAPGothic'
def split_function_now(a_s,a_e,a_1,a_2,a_4,inf_year,time2):

    # time２が年単位なのでd2s(day to second)と365をかけて元の単位に戻す
    time = (time2 - time2[0])*data.cfg.d2s*365
    T_s = 0.0
    T_e = inf_year*365*60*60*24 # 厳密には

    # 最適化区間の倍をとる（大スケールの再現を行うため）
    omega = 2*np.pi/(inf_year*365*60*60*24*2)

    sin1 = a_1*np.sin(1.0*omega*time)
    sin2 = a_2*np.sin(2.0*omega*time)
    sin4 = a_4*np.sin(4.0*omega*time)
    # lin  = (a_s*(time[-1]-time)+a_e*(time-time[0]))/(time[-1]-time[0])
    lin  = (a_s*(T_e-time)+a_e*(time-T_s))/(T_e-T_s)
    plt.plot(time,lin+sin1+sin2+sin4,'g',linewidth=3,label='元関数')
    plt.plot(time,lin+sin1,'r--',label='1次成分')
    plt.plot(time,lin+sin2,'m--',label='2次成分')
    plt.plot(time,lin+sin4,'c--',label='4次成分')
    plt.plot(time,lin,'k--',label='線形成分')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=1.6*size, loc='upper left')  # 凡例を右上に固定
    plt.tight_layout()
    plt.savefig("split_functions_now.png", dpi=300)  # 解像度を高める
    plt.clf()
split_function_now(a_s,a_e,a_1,a_2,a_4,inf_year,time2)
# ============================================================================== #
cc   = np.sum((SN1-np.mean(SN1))*(SN2-np.mean(SN2)))/np.sqrt(np.sum((SN1-np.mean(SN1))**2)*np.sum((SN2-np.mean(SN2))**2))
sd = np.sum((SN1 - SN2)**2) / np.sum(SN1**2)
print("----------------------------------------------")
print("相関係数＝",cc,"NMSE=",sd)
print("評価関数=",alpha*cc-(1-alpha)*sd)
