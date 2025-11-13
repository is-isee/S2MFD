import matplotlib.pyplot as plt
import numpy as np
import os, sys
sys.path.append('../')
import S2MFD

# 分析範囲、対象は手で決める
datadir1 = '../num_results/data_ground_no2/'
datadir2 = '../OBS_results_no4/data_gt_u0_no2/'


n0 = 2266
n1 = 2768
alpha = 0.9
data = S2MFD.Data.initial_load(datadir1)

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
# datadir1
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
        
SN1 = Karak_deffine(Bpht,base,cfg,grid)
so0t1 = so0t
uu0t1 = uu0t
time1 = timet/data.cfg.d2s/365
# ============================================================================== #
# datadir2
data = S2MFD.Data.initial_load(datadir2)

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
    d = np.load(file=datadir2+'data.'+str(n).zfill(6)+'.npz')
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
# 描画のための設定
fontsize = 35

plt.close('all')
plt.clf()
fig = plt.figure('compare_hotta',figsize=(10,18))
ax1 = fig.add_subplot(3,1,1)
ax2 = fig.add_subplot(3,1,2)
ax3 = fig.add_subplot(3,1,3)

# 日本語使用可能
import matplotlib as mpl
mpl.rcParams['font.family'] = 'IPAPGothic'
plt.rcParams['font.size'] = fontsize  # グローバルなフォントサイズ設定
# グラフの描画
# plt.figure(figsize=(10, 6))  # グラフのサイズを調整
ax1.plot(time1, SN1, 'r', label='ground truth')  # ラベル名を明確に
ax1.plot(time2, SN2, 'b', label='GA inference')  # ラベル名を明確に
# ax1.plot(time1, SN1, 'r')  # ラベル名を明確に
# ax1.plot(time2, SN2, 'b')  # ラベル名を明確に
# plt.plot(time1, SN1, 'r--',linewidth=2.5)  # ラベル名を明確に
# plt.plot(time2, SN2, 'b')  # ラベル名を明確に
# 軸ラベル
# plt.xlabel('年', fontsize=3*size)
# plt.ylabel('黒点相対数', fontsize=3*size)
# plt.title('黒点数時間変化', fontsize=4*size)  # タイトルを追加
# ax1.set_xlabel(r'$t~[\rm{yr}]$')
ax1.set_ylabel('Sunspot number',fontsize=40)
# 軸のメモリを細かく設定
# ax1.tick_params(labelsize=40)  # 目盛りの文字サイズを20に
# plt.xticks(fontsize=2*size)  # x軸の数値サイズを調整
# plt.yticks(fontsize=2*size)  # y軸の数値サイズを調整
# グリッドを追加して見やすく
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.set_ylim(0, np.max([SN1,SN2])*1.1)
# # 凡例を表示
# ax1.legend(fontsize=25, loc='upper left',frameon=False)  # 凡例を右上に固定
ax1.set_title('(a)')
# グラフを保存


# plt.clf()



ax2.plot(time1,uu0t1,'r')
ax2.plot(time2,uu0t2,'b')
# plt.xlabel('年',fontsize=3*size)
# plt.xlabel(r'$t~[\rm{yr}]$', fontsize=2.5*size)
ax2.set_ylabel(r'$u_0~[\rm{cm~s^{-1}}]$',fontsize=50)
# plt.title(r'$u_0$ 時間変化', fontsize=4*size)  # タイトルを追加
ymax = max(np.max(uu0t1), np.max(uu0t2)) * 1.1
ax2.set_ylim(bottom=0, top=ymax)
# 軸のメモリを細かく設定
# plt.xticks(fontsize=2*size)  # x軸の数値サイズを調整
# plt.yticks(fontsize=2*size)  # y軸の数値サイズを調整
# グリッドを追加して見やすく
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.set_title('(c)')
# 凡例を表示
# plt.legend(fontsize=1.6*size, loc='upper right')  # 凡例を右上に固定
# グラフを保存
# plt.savefig("P_u0_compare.png", dpi=300)

ax3.plot(time1,so0t1,'r')
ax3.plot(time2,so0t2,'b')
ymax = max(np.max(so0t1), np.max(so0t2)) * 1.1
ax3.set_ylim(bottom=0, top=ymax)
# plt.xlabel('年',fontsize=3*size)
ax3.set_xlabel(r'$t~[\rm{yr}]$', fontsize=50)
ax3.set_ylabel(r'$s_0~[\rm{cm~s^{-1}}]$',fontsize=50)
# plt.title(r'$s_0$ 時間変化', fontsize=4*size)  # タイトルを追加
# 軸のメモリを細かく設定
# ax3.xticks(fontsize=2*size)  # x軸の数値サイズを調整
# ax3.yticks(fontsize=2*size)  # y軸の数値サイズを調整
# グリッドを追加して見やすく
ax3.grid(True, linestyle='--', alpha=0.7)
ax3.set_title('(e)')
# 凡例を表示
# plt.legend(fontsize=1.6*size, loc='upper right')  # 凡例を右上に固定
# グラフを保存
# plt.savefig("P_s0_compare.png", dpi=300)
# plt.clf()

fig.tight_layout()
fig.savefig("P_sunspots_number_compare.png", dpi=300)  # 解像度を高める

import sys
# sys.exit()

# ============================================================================== #
cc   = np.sum((SN1-np.mean(SN1))*(SN2-np.mean(SN2)))/np.sqrt(np.sum((SN1-np.mean(SN1))**2)*np.sum((SN2-np.mean(SN2))**2))
sd = np.sum((SN1 - SN2)**2) / np.sum(SN1**2)
# so0t_dif = (so0t2/so0t1).mean()
# uu0t_dif = (uu0t2/uu0t1).mean()
uu0t_mpe = (abs((uu0t2-uu0t1)/uu0t1)).mean()*100
so0t_mpe = (abs((so0t2-so0t1)/so0t1)).mean()*100
uu0tt_nmse = np.sum((uu0t1 - uu0t2)**2) / np.sum(uu0t1**2)
so0tt_nmse = np.sum((so0t1 - so0t2)**2) / np.sum(so0t1**2)

print("----------------------------------------------")
print("r_sunspot=",cc,"e_sunspot",sd)
print("評価関数=",alpha*cc-(1-alpha)*sd)
# print("uu0の比率=",uu0t_dif)
# print("so0の比率=",so0t_dif)
# print("uu0の誤差=",100*abs(1-uu0t_dif))
# print("so0の誤差=",100*abs(1-so0t_dif))
# print("u0_nmse=",uu0tt_nmse)
# print("s0_nmse=",so0tt_nmse)
print("----------発表資料には以下を使用----------")
print("uu0のMPE＝",uu0t_mpe)
print("so0のMPE＝",so0t_mpe)
