import matplotlib.pyplot as plt
import numpy as np
import os, sys
sys.path.append('../')
import S2MFD

# 分析範囲、対象は手で決める
datadir1 = '../data_sin_GA/data_test_s0u0/'
datadir2 = '../data_sin_GA/data_GA_s0u0_2/'
n0 = 1040
n1 = 1610

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
plt.plot(time1,SN1,'r',label = 'sunspots number')
plt.plot(time2,SN2,'b',label = 'sunspots number')
plt.xlabel('time(year)',fontsize=20)
plt.ylabel('sunspots number',fontsize=20)
plt.savefig("P_sunspots_number.png")
plt.clf()

plt.plot(time1,uu0t1,'r',label = 'meridional flow speed')
plt.plot(time2,uu0t2,'b',label = 'meridional flow speed')
plt.xlabel('time(year)',fontsize=20)
plt.ylabel('meridional flow speed',fontsize=20)
plt.savefig("P_u0.png")
plt.clf()

plt.plot(time1,so0t1,'r',label = 'alpha effect')
plt.plot(time2,so0t2,'b',label = 'alpha effect')
plt.xlabel('time(year)',fontsize=20)
plt.ylabel('s_0',fontsize=20)
plt.savefig("P_s0.png")
plt.clf()
# ============================================================================== #
cc   = np.sum((SN1-np.mean(SN1))*(SN2-np.mean(SN2)))/np.sqrt(np.sum((SN1-np.mean(SN1))**2)*np.sum((SN2-np.mean(SN2))**2))
sd   = np.sqrt((np.sum(SN1) - np.sum(SN2))**2) / np.sum(SN1)
so0t_dif = (so0t1/so0t2).mean()
uu0t_dif = (uu0t1/uu0t2).mean()

print("相関係数＝",cc,"誤差割合＝",sd)
print("so0の比=",so0t_dif)
print("uu0の比=",uu0t_dif)
