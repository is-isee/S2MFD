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
ns = 1569
ne = 2100
time_y = timet/data.cfg.d2s/365
B_range = 0.05
plt.clf()
plt.close('all')
fig = plt.figure('Butterfly Diagram',figsize=(10,5))
ax = fig.add_subplot(1,1,1)

# 描画範囲指定→年数で指定
# ax1.set_xlim(145,190)
# ax2.set_xlim(145,190)
# butterfly diagram
B_0 = 1

# 軸ラベルなどのフォントサイズを一気に指定
size=8

# Hotta+2010の場合
# B_0 = 4.e4
c1 = ax.pcolormesh(time_y[ns:ne], grid.th/np.pi*180, B_0*Bpht_c[:,ns:ne], cmap='bwr', shading='auto')
c2 = ax.contour(time_y[ns:ne],grid.th/np.pi*180,B_0*Brrt_s[:,ns:ne],vmax=B_range,vmin=-B_range,colors='black',levels=np.linspace(-0.02,0.02,10))
# c2 = ax2.pcolormesh(time_y[ns:ne],grid.th/np.pi*180,B_0*Brrt_s[:,ns:ne], cmap='bwr',shading='auto')
# 描画範囲で色を綺麗にしたい場合
# c1 = ax1.pcolormesh(time_y[ns:ne], grid.th/np.pi*180, B_0*Bpht_c[:,ns:ne],vmax=B_0*np.max(Bpht_c[:,2000:2500]),vmin=-B_0*np.max(Bpht_c[:,2000:2500]), cmap='bwr', shading='auto')
# 描画の範囲指定なし
# c2 = ax2.pcolormesh(time_y[ns:ne], grid.th/np.pi*180, B_0*Brrt_s[:,ns:ne], cmap='bwr', shading='auto')

# カラーバーを追加
fig.colorbar(c1, ax=ax, orientation='vertical').set_label(r'$B_\phi$', fontsize=3*size)
# fig.colorbar(c2, ax=ax2, orientation='vertical').set_label(r'$B_r$ (G)', fontsize=20)
# cbar1 = fig.colorbar(c1, ax=ax1, orientation='vertical')
# cbar1.set_label(r'$B_\phi$', fontsize=3*size)
# cbar1.ax.tick_params(labelsize=1.5*size)  # ← カラーバーの数字のフォントサイズ

# cbar2 = fig.colorbar(c2, ax=ax2, orientation='vertical')
# cbar2.set_label(r'$B_r$', fontsize=3*size)
# cbar2.ax.tick_params(labelsize=1.5*size)  # ← カラーバーの数字のフォントサイズ

# sunspot
Bpht_lim = Bpht_c[:,ns:ne]
mask_p = Bpht_lim > 3.0
mask_m = Bpht_lim < -3.0
# sunspotを描画
# ax1.contourf(time_y[ns:ne], grid.th/np.pi*180, mask_p, levels=[0.5, 1.5], colors=['black'])
# ax1.contourf(time_y[ns:ne], grid.th/np.pi*180, mask_m, levels=[0.5, 1.5], colors=['black'])

# ラベル名・サイズの指定
import matplotlib as mpl
mpl.rcParams['font.family'] = 'IPAPGothic'
# ax1.set_title("蝶形図", fontsize=4*size)
ax.set_xlabel(r'$t~[\rm{yr}]$',fontsize=3*size)
ax.set_ylabel('Latitude [degree]',fontsize=2*size)
ax.tick_params(axis='both', which='major', labelsize=15)
plt.tight_layout()
plt.savefig("P_butterfly.png")
plt.clf()
plt.close('all')

