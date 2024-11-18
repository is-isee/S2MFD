import matplotlib.pyplot as plt
import numpy as np
import os, sys
sys.path.append('../')
import S2MFD

datadir = '../data/'
data = S2MFD.S2MFD_data.initial_load(datadir)

cfg = data.cfg
grid = data.grid
setup = data.setup

n1 = 0
if os.path.isdir(datadir):
    # dataディレクトリ内の最も大きな番号を探る
    # 特定のステップから始めたい場合は、そのステップを手で指定する
    files = os.listdir(datadir)
    for file in files:
        filel = file.split('.')
        if filel[0] == 'data':
            n1 = max(n1, int(filel[1]))
            
n0 = 0
tau_diff = data.cfg.RSUN**2/data.cfg.ett
timet = np.zeros(n1-n0)
Brrt = np.zeros((grid.ixg,grid.jxg,n1-n0))
Btht = np.zeros((grid.ixg,grid.jxg,n1-n0))
Bpht = np.zeros((grid.ixg,grid.jxg,n1-n0))
for n  in range(n0,n1):
    data.data_load(n)
    Brr, Bth = S2MFD.poloidal_mag(data.Aph, grid.RR, grid.sinTH, grid.drr, grid.dth)
    d = np.load(file=datadir+'data.'+str(n).zfill(6)+'.npz')
    timet[n-n0] = d['time']
    Brrt[:,:,n-n0] = Brr
    Btht[:,:,n-n0] = Bth
    Bpht[:,:,n-n0] = d['Bph']
    
#
Bpht0 = Bpht[1+np.argmin(abs(grid.rr-0.7*cfg.RSUN)),np.argmin(abs(grid.th-30/180*np.pi)),:]
Brrt0 = Brrt[-2,np.argmin(abs(grid.th-60/180*np.pi)),:]

Bpht0_sign = np.sign(Bpht0)
Bpht0_sign_diff = np.diff(Bpht0_sign)

ns = np.where(Bpht0_sign_diff == +2)[0][-2]
ne = np.where(Bpht0_sign_diff == +2)[0][-1]

plt.clf()
fig = plt.figure('J08_test',figsize=(15,10))
ax1 = fig.add_subplot(2,3,1)
ax2 = fig.add_subplot(2,3,2)
ax3 = fig.add_subplot(2,3,3)
ax4 = fig.add_subplot(2,3,4)
ax5 = fig.add_subplot(2,3,5)
ax6 = fig.add_subplot(2,3,6)

timeu = (timet[ns:ne]-timet[ns])/tau_diff
Bpht0u = Bpht0[ns:ne]
Brrt0u = Brrt0[ns:ne]
nw = ne - ns
nm = np.argmax(Bpht0u)

ax1.plot(timeu,Bpht0u)
ax2.plot(timeu[nm-nw//20:nm+nw//20],Bpht0u[nm-nw//20:nm+nw//20])
ax3.plot(timeu[nw-nw//10:nw],Bpht0u[nw-nw//10:nw])

ax4.plot(timeu,Brrt0u)

    

