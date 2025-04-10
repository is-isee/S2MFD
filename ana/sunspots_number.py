import matplotlib.pyplot as plt
import numpy as np
import os, sys
sys.path.append('../')
import S2MFD


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

n_conv = 4 #移動平均の個数
conv_f = np.ones(n_conv)/n_conv
S_num2 = np.convolve(S_num, conv_f, mode='same')#移動平均
N_num2 = np.convolve(N_num, conv_f, mode='same')#移動平均

# time_s = np.linspace(0,data.cfg.tend,data.cfg.tend//data.cfg.dtout)
time_s = timet
time_y = time_s/data.cfg.d2s/365
plt.plot(time_y,S_num2,'r',label = 'sunspots number')
# plt.xlim(200, 300)
plt.xlabel('time(year)')
plt.ylabel('sunspots number')
plt.savefig("sunspots_number.png")
plt.clf()

plt.plot(time_y,uu0t,'r',label = 'meridional flow speed')
plt.xlabel('time(year)')
plt.ylabel('meridional flow speed')
plt.savefig("meridional_flow_speed.png")
plt.clf()

plt.plot(time_y,so0t,'r',label = 'alpha effect')
plt.xlabel('time(year)')
plt.ylabel('alpha effect')
plt.savefig("alpha_effect.png")
plt.clf()