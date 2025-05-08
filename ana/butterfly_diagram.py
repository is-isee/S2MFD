Bpht_c = Bpht[1+np.argmin(abs(grid.rr-0.7*cfg.RSUN)),:,:]
Brrt_s = Brrt[-2,:,:]


Bpht0 = Bpht[1+np.argmin(abs(grid.rr-0.7*cfg.RSUN)),np.argmin(abs(grid.th-30/180*np.pi)),:]
Brrt0 = Brrt[-2,np.argmin(abs(grid.th-60/180*np.pi)),:]

Bpht0_sign = np.sign(Bpht0)
Bpht0_sign_diff = np.diff(Bpht0_sign)
"""
# 拡散時間等を横軸にしたくなった場合
ns = np.where(Bpht0_sign_diff == +2)[0][-3]
ne = np.where(Bpht0_sign_diff == +2)[0][-1]
ns = 0
ne = 4999
timeu = (timet[ns:ne]-timet[ns])/tau_diff
time_year = (timet[ns:ne]-timet[ns])/86400/365
"""
ns = 0
ne = 5000
time_y = timet/data.cfg.d2s/365
B_range = 3
plt.clf()
plt.close('all')
fig = plt.figure('Butterfly Diagram',figsize=(10,10))
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)

# butterfly diagram
B_0 = 1.e5
# Hotta+2010の場合
# B_0 = 4.e4
c1 = ax1.pcolormesh(time_y[ns:ne], grid.th/np.pi*180, B_0*Bpht_c[:,ns:ne], cmap='bwr', shading='auto')
c2 = ax2.pcolormesh(time_y[ns:ne],grid.th/np.pi*180,B_0*Brrt_s[:,ns:ne],vmax=B_range*1e3,vmin=-B_range*1e3,cmap='bwr',shading='auto')
# 描画の範囲指定なし
# c2 = ax2.pcolormesh(time_y[ns:ne], grid.th/np.pi*180, B_0*Brrt_s[:,ns:ne], cmap='bwr', shading='auto')

# カラーバーを追加
fig.colorbar(c1, ax=ax1, orientation='vertical').set_label(r'$B_\phi$ (G)', fontsize=20)
fig.colorbar(c2, ax=ax2, orientation='vertical').set_label(r'$B_r$ (G)', fontsize=20)

# sunspot
Bpht_lim = Bpht_c[:,ns:ne]
mask_p = Bpht_lim > 3.0
mask_m = Bpht_lim < -3.0
ax1.contourf(time_y[ns:ne], grid.th/np.pi*180, mask_p, levels=[0.5, 1.5], colors=['black'])
ax1.contourf(time_y[ns:ne], grid.th/np.pi*180, mask_m, levels=[0.5, 1.5], colors=['black'])

ax1.set_ylabel(r'$B_\phi$: $r=0.7R_\odot$',fontsize=20)
ax2.set_ylabel(r'$B_r$: $r=R_\odot$',fontsize=20)

ax2.set_xlabel('t(year)',fontsize=20)
plt.savefig("P_butterfly.png")
plt.clf()
plt.close('all')

