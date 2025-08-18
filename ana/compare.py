import matplotlib.pyplot as plt
import numpy as np
import os, sys
sys.path.append('../')
import S2MFD

# 分析範囲、対象は手で決める
datadir1 = '../data_sinsamp_u0s0/'
datadir2 = '../data_num_u0/'


n0 = 958
n1 = 1355
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
# グラフの描画
plt.figure(figsize=(10, 6))  # グラフのサイズを調整
plt.plot(time1, SN1, 'r--', label='Observation',linewidth=2.5)  # ラベル名を明確に
plt.plot(time2, SN2, 'b', label='GA inference')  # ラベル名を明確に
# 軸ラベル
plt.xlabel('Years', fontsize=20)
plt.ylabel('Sunspots Number', fontsize=20)
# 軸のメモリを細かく設定
plt.xticks(fontsize=14)  # x軸の数値サイズを調整
plt.yticks(fontsize=14)  # y軸の数値サイズを調整
# グリッドを追加して見やすく
plt.grid(True, linestyle='--', alpha=0.7)
# 凡例を表示
plt.legend(fontsize=14, loc='upper right')  # 凡例を右上に固定
# グラフを保存
plt.savefig("P_sunspots_number_compare.png", dpi=300)  # 解像度を高める
plt.clf()

plt.plot(time1,uu0t1,'r--',label='Observation',linewidth=2.5)
plt.plot(time2,uu0t2,'b',label='GA inference')
plt.xlabel('Years',fontsize=20)
plt.ylabel(r'$u_0(\rm{cm/s})$',fontsize=20)
ymax = max(np.max(uu0t1), np.max(uu0t2)) * 1.1
plt.ylim(bottom=0, top=ymax)
# 軸のメモリを細かく設定
plt.xticks(fontsize=14)  # x軸の数値サイズを調整
plt.yticks(fontsize=14)  # y軸の数値サイズを調整
# グリッドを追加して見やすく
plt.grid(True, linestyle='--', alpha=0.7)
# 凡例を表示
plt.legend(fontsize=14, loc='upper right')  # 凡例を右上に固定
# グラフを保存
plt.savefig("P_u0_compare.png", dpi=300)
plt.clf()

plt.plot(time1,so0t1,'r--',label='Observation',linewidth=2.5)
plt.plot(time2,so0t2,'b',label='GA inference')
ymax = max(np.max(so0t1), np.max(so0t2)) * 1.1
plt.ylim(bottom=0, top=ymax)
plt.xlabel('Years',fontsize=20)
plt.ylabel(r'$s_0(\rm{cm/s})$',fontsize=20)
# 軸のメモリを細かく設定
plt.xticks(fontsize=14)  # x軸の数値サイズを調整
plt.yticks(fontsize=14)  # y軸の数値サイズを調整
# グリッドを追加して見やすく
plt.grid(True, linestyle='--', alpha=0.7)
# 凡例を表示
plt.legend(fontsize=14, loc='upper right')  # 凡例を右上に固定
# グラフを保存
plt.savefig("P_s0_compare.png", dpi=300)
plt.clf()

# ============================================================================== #
cc   = np.sum((SN1-np.mean(SN1))*(SN2-np.mean(SN2)))/np.sqrt(np.sum((SN1-np.mean(SN1))**2)*np.sum((SN2-np.mean(SN2))**2))
sd   = np.sqrt((np.sum(SN1) - np.sum(SN2))**2) / np.sum(SN1)
so0t_dif = (so0t2/so0t1).mean()
uu0t_dif = (uu0t2/uu0t1).mean()
print("----------------------------------------------")
print("相関係数＝",cc,"誤差割合＝",sd)
print("評価関数=",alpha*cc-(1-alpha)*sd)
print("so0の比=",so0t_dif)
print("uu0の比=",uu0t_dif)
