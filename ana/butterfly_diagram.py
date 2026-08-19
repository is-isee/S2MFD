"""蝶形図 (butterfly diagram) を描くスクリプト。

使い方:
    python butterfly_diagram.py [datadir]   # 既定は ../data/
"""
import matplotlib.pyplot as plt
import numpy as np

from ana_common import get_datadir, load_run

datadir = get_datadir()
run = load_run(datadir)
cfg, grid, timet, tau_diff = run.cfg, run.grid, run.timet, run.tau_diff
Bpht, Brrt = run.Bpht, run.Brrt

Bpht_c = Bpht[1+np.argmin(abs(grid.rr-0.7*cfg.RSUN)),:,:]
Brrt_s = Brrt[-2,:,:]


Bpht0 = Bpht[1+np.argmin(abs(grid.rr-0.7*cfg.RSUN)),np.argmin(abs(grid.th-30/180*np.pi)),:]
Brrt0 = Brrt[-2,np.argmin(abs(grid.th-60/180*np.pi)),:]

Bpht0_sign = np.sign(Bpht0)
Bpht0_sign_diff = np.diff(Bpht0_sign)

ns = np.where(Bpht0_sign_diff == +2)[0][-3]
ne = np.where(Bpht0_sign_diff == +2)[0][-1]

timeu = (timet[ns:ne]-timet[ns])/tau_diff
time_year = (timet[ns:ne]-timet[ns])/86400/365
plt.clf()
plt.close('all')
fig = plt.figure('Butterfly Diagram',figsize=(10,10))
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)

# butterfly diagram
# B_0: 磁場の規格化 (クエンチング B/(1+B^2) の単位磁場) [G]
B_0 = 4.e4
c1 = ax1.pcolormesh(time_year, grid.th/np.pi*180, B_0*Bpht_c[:,ns:ne], cmap='bwr', shading='auto')
c2 = ax2.pcolormesh(time_year, grid.th/np.pi*180, B_0*Brrt_s[:,ns:ne], cmap='bwr', shading='auto')

# カラーバーを追加
fig.colorbar(c1, ax=ax1, orientation='vertical').set_label(r'$B_\phi$ (G)')
fig.colorbar(c2, ax=ax2, orientation='vertical').set_label(r'$B_r$ (G)')

# sunspot
Bpht_lim = Bpht_c[:,ns:ne]
mask_p = Bpht_lim > 1.0
mask_m = Bpht_lim < -1.0
ax1.contourf(timeu, grid.th/np.pi*180, mask_p, levels=[0.5, 1.5], colors=['black'])
ax1.contourf(timeu, grid.th/np.pi*180, mask_m, levels=[0.5, 1.5], colors=['black'])

ax1.set_ylabel(r'$B_\phi$: $r=0.7R_\odot$')
ax2.set_ylabel(r'$B_r$: $r=R_\odot$')

ax2.set_xlabel('t(year)')

fig.tight_layout()
plt.show()
