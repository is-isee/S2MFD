import matplotlib.pyplot as plt
import numpy as np
import os, sys
sys.path.append('../')
import S2MFD

"""
# 黒点数の計上
N_num = np.zeros(n1)
S_num = np.zeros(n1)
thrsh = 3.0
base  = 1+np.argmin(abs(grid.rr-0.7*cfg.RSUN))
stat  = 1+np.argmin(abs(grid.th- 40/180*np.pi))
endd  = 1+np.argmin(abs(grid.th-140/180*np.pi))
# グリッドに応じた北半球と南半球の分割
if grid.jxg % 2==0: 
    S_equa = grid.jxg//2 - 1
    N_equa = S_equa + 1
else:
    S_equa = (grid.jxg-1)//2 - 1
    N_equa = S_equa + 2

# S_numとN_numは全く同じになる（南北対称だから当たり前）
# n1は時間要素の最後の番号
for t in range(n1):
    # S_num[t] = np.sum(Bpht[base,grid.margin:S_equa+1,t]<-thrsh) + np.sum(Bpht[base,grid.margin:S_equa+1,t]>thrsh)
    # N_num[t] = np.sum(Bpht[base,N_equa:grid.jxg-grid.margin,t]<-thrsh) + np.sum(Bpht[base,N_equa:grid.jxg-grid.margin,t]>thrsh)
    S_num[t] = np.sum(Bpht[base,stat:S_equa+1,t]<-thrsh) + np.sum(Bpht[base,stat:S_equa+1,t]>thrsh)
    N_num[t] = np.sum(Bpht[base,N_equa:endd,t]<-thrsh)   + np.sum(Bpht[base,N_equa:endd,t]>thrsh)
# 観測に基づいた調整パラメタ
kappa = 0.3
S_num = kappa * S_num
N_num = kappa * N_num

# 移動平均関数
def moving_average(SunspotsNum, n_conv):
    conv_f = np.ones(n_conv)/n_conv
    SN_smooth = np.convolve(SunspotsNum, conv_f, mode='same')#移動平均
    
    return SN_smooth

S_num2 = moving_average(S_num, 4)
N_num2 = moving_average(N_num, 4)
"""

base  = 1+np.argmin(abs(grid.rr-0.7*cfg.RSUN))
# 移動平均関数
def moving_average(SunspotsNum, n_conv):
    conv_f = np.ones(n_conv)/n_conv
    SN_smooth = np.convolve(SunspotsNum, conv_f, mode='same')#移動平均
    
    return SN_smooth
# 磁場エネルギー密度B^2(toroidal,15°,r_c)を黒点数の指標とする。
def Karak_deffine(Bpht,base,cfg,grid):
    SN = Bpht[base,np.argmin(abs(grid.th-75/180*np.pi)),:]**2
    SN2 = moving_average(SN, 4)
    return SN

# 磁場エネルギー密度B^2(toroidal,10°~20°,r_c)を黒点数の指標とする。
def original_deffine(Bpht,base,cfg,grid):
    locap =   np.argmin(abs(grid.th- 80/180*np.pi))
    locam =   np.argmin(abs(grid.th- 70/180*np.pi))
    SN = np.mean(Bpht[base,locam:locap,:]**2,axis=0)
    SN2 = moving_average(SN, 4)
    return SN2

# 関数の実行
# SN2 = original_deffine(Bpht,base,cfg,grid)
SN2 = Karak_deffine(Bpht,base,cfg,grid)

# 時間の配列生成
# time_s = np.linspace(0,data.cfg.tend,data.cfg.tend//data.cfg.dtout)
time_s = timet
time_y = time_s/data.cfg.d2s/365

# グラフの描画
# plt.plot(time_y,S_num2,'r',label = 'sunspots number')
# Sunspots Number
plt.figure(figsize=(10, 6))  # グラフのサイズを調整
# plt.xlim(110,500)  # x軸の範囲を調整
# plt.ylim(0,2)  # x軸の範囲を調整
plt.plot(time_y, SN2, 'r', label='Sunspots Number')  # ラベル名を明確に
plt.xlabel('Years', fontsize=20)
plt.ylabel('Sunspots Number', fontsize=20)
plt.xticks(fontsize=14)  # x軸の数値サイズを調整
plt.yticks(fontsize=14)  # y軸の数値サイズを調整
plt.grid(True, linestyle='--', alpha=0.7)  # グリッドを追加
plt.legend(fontsize=14, loc='upper right')
plt.savefig("P_sunspots_number.png", dpi=300)  # 解像度を高める
plt.clf()

# Meridional Flow Speed
plt.figure(figsize=(10, 6))  # グラフのサイズを調整
plt.plot(time_y, uu0t, 'r', label='Meridional Flow Speed')  # ラベル名を明確に
plt.xlabel('Years', fontsize=20)
plt.ylabel(r'$u_0(\rm{cm/s})$', fontsize=20)  # LaTeX形式で表示
plt.xticks(fontsize=14)  # x軸の数値サイズを調整
plt.yticks(fontsize=14)  # y軸の数値サイズを調整
plt.grid(True, linestyle='--', alpha=0.7)  # グリッドを追加
plt.legend(fontsize=14, loc='upper right')
plt.savefig("P_u0.png", dpi=300)  # 解像度を高める
plt.clf()

# Alpha Effect
plt.figure(figsize=(10, 6))  # グラフのサイズを調整
plt.plot(time_y, so0t, 'r', label='Alpha Effect')  # ラベル名を明確に
plt.xlabel('Years', fontsize=20)
plt.ylabel(r'$s_0(\rm{cm/s})$', fontsize=20)  # LaTeX形式で表示
plt.xticks(fontsize=14)  # x軸の数値サイズを調整
plt.yticks(fontsize=14)  # y軸の数値サイズを調整
plt.grid(True, linestyle='--', alpha=0.7)  # グリッドを追加
plt.legend(fontsize=14, loc='upper right')
plt.savefig("P_s0.png", dpi=300)  # 解像度を高める
plt.clf()