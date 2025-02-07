import matplotlib.pyplot as plt
import numpy as np
import os, sys
sys.path.append('../')
import S2MFD

# datadir = '../data_alpha_omega_etaconst/'
datadir = '../data_alpha_omega/'
# datadir = '../data_flux_transport/'
# datadir = '../data_potential/'
data = S2MFD.Data.initial_load(datadir)

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
    print(n)
    data.data_load(n)
    Brr, Bth = S2MFD.physics.poloidal_mag(data.Aph, grid.RR, grid.sinTH, grid.drr, grid.dth)
    d = np.load(file=datadir+'data.'+str(n).zfill(6)+'.npz')
    timet[n-n0] = d['time']
    Brrt[:,:,n-n0] = Brr
    Btht[:,:,n-n0] = Bth
    Bpht[:,:,n-n0] = d['Bph']
    
#
Bpht_7 = Bpht[1+np.argmin(abs(grid.rr-0.7*cfg.RSUN)),:,:]
Brrt_s = Brrt[-2,:,:]


Bpht0 = Bpht[1+np.argmin(abs(grid.rr-0.7*cfg.RSUN)),np.argmin(abs(grid.th-30/180*np.pi)),:]
Brrt0 = Brrt[-2,np.argmin(abs(grid.th-60/180*np.pi)),:]

Bpht0_sign = np.sign(Bpht0)
Bpht0_sign_diff = np.diff(Bpht0_sign)

ns = np.where(Bpht0_sign_diff == +2)[0][-2]
ne = np.where(Bpht0_sign_diff == +2)[0][-1]
# timeu = (timet[ns:ne]-timet[ns])/tau_diff
timeu = (timet[ns:]-timet[ns])/tau_diff
B_range = 3
plt.clf()
plt.close('all')
fig = plt.figure('Butterfly Diagram',figsize=(6,10))
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)
# Bpht0u = Bpht0[ns:ne]
# Brrt0u = Brrt0[ns:ne]
# nw = ne - ns
# nm = np.argmax(Bpht0u)
ax1.pcolormesh(timeu,grid.th/np.pi*180,Bpht_7[:,ns:],cmap='bwr',shading='auto')
ax2.pcolormesh(timeu,grid.th/np.pi*180,Brrt_s[:,ns:],vmax=B_range*1e-2,vmin=-B_range*1e-2,cmap='bwr',shading='auto')

ax1.set_ylabel(r'$B_\phi$: $r=0.7R_\odot$')
ax2.set_ylabel(r'$B_r$: $r=R_\odot$')

ax2.set_xlabel(r't/$\tau_\mathrm{diff}$')

fig.tight_layout()