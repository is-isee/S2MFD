sys.path.append('../')
import S2MFD

# 確認したいパターンを自分で指定
cfg = S2MFD.Cfg('parameters/parameter_sample.py')
grid = S2MFD.Grid(ix=cfg.ix, jx=cfg.jx, margin=cfg.margin, rrmin=cfg.rrmin, rrmax=cfg.rrmax, thmin=cfg.thmin, thmax=cfg.thmax)
setup = S2MFD.Setup(cfg,grid)

# Differencial Rotation
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
contour_1 = ax.contourf(np.pi/2-grid.TH, grid.RR/cfg.RSUN, setup.om, 100, cmap='viridis')
contour_2 = ax.contour(np.pi/2-grid.TH, grid.RR/cfg.RSUN, setup.om, levels=14, linewidths=1,colors='black')
cbar = plt.colorbar(contour_1)
cbar.set_label('Differential rotation')
plt.title("Differential rotation")
ax.set_xlim(0, np.pi/2)
ax.set_ylim(0,1.0)
plt.savefig("P_differencial.png")
plt.clf()
plt.close('all')

# Meridional
U_x = setup.urr * grid.cosTH - setup.uth * grid.sinTH
U_y = setup.urr * grid.sinTH + setup.uth * grid.cosTH
fig, ax = plt.subplots()
plt.quiver(grid.Y, grid.X, U_y, U_x)
# plt.streamplot(grid.RR, grid.RR, U_y, U_x)
plt.title('The Flow Fields')
ax.set_aspect('equal')
plt.xlabel('length(cm)')
plt.ylabel('length(cm)')
plt.title("Meridional Flow")
plt.ylim(-7e10,7e10)
plt.xlim(0,7e10)
plt.savefig("P_meridional.png")
plt.clf()
plt.close('all')

# Diffusivity
plt.figure(figsize=(6, 6))  # 描画領域を正方形にする
plt.xlabel('$r/R$')
plt.ylabel('$\eta{(cm^2s^{-1})}$')
plt.ylim(1e8,1e13)
plt.xlim(np.min(grid.RR)/cfg.RSUN,np.max(grid.RR)/cfg.RSUN)
plt.plot(grid.RR[:,1]/cfg.RSUN,setup.et[:,1])
ax.set_aspect('equal')
plt.yscale('log')
plt.title("Diffusivity")
ax.set_box_aspect(1)  # 縦横比を1:1に設定 (Matplotlib v3.3+)
plt.savefig("P_diffusivity.png")
plt.clf()
plt.close('all')

# alpha (θ＝60)
plt.figure(figsize=(6, 6))  # 描画領域を正方形にする
plt.plot(grid.RR/cfg.RSUN,setup.so[:,43])
ax.set_aspect('equal')
ax.set_box_aspect(1)  # 縦横比を1:1に設定 (Matplotlib v3.3+)
plt.xlabel('$r/R$')
plt.ylabel(r'$\alpha{(cm s^{-1})}$')
plt.xlim(np.min(grid.RR)/cfg.RSUN,np.max(grid.RR)/cfg.RSUN)
plt.title("alpha effect(θ=60)")
plt.savefig("P_alpha(θ=60).png")
plt.clf()
plt.close('all')

# alpha (r=0.975R)
plt.figure(figsize=(6, 6))  # 描画領域を正方形にする
plt.plot(grid.th*180/np.pi,setup.so[127,:])
ax.set_aspect('equal')
ax.set_box_aspect(1)  # 縦横比を1:1に設定 (Matplotlib v3.3+)
plt.xlabel(r'$\theta$')
plt.ylabel(r'$\alpha{(cm s^{-1})}$')
plt.xlim(0,np.max(grid.th*180/np.pi))
plt.title("alpha effect(r=0.975R)")
plt.savefig("P_alpha(r=0.975R).png")
plt.clf()
plt.close('all')

# alpha contour
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
contour_1 = ax.contourf(np.pi/2-grid.TH, grid.RR/cfg.RSUN, setup.so, 100, cmap='viridis')
contour_2 = ax.contour(np.pi/2-grid.TH, grid.RR/cfg.RSUN, setup.so, levels=14, linewidths=1,colors='black')
cbar = plt.colorbar(contour_1)
cbar.set_label('source term',fontsize=20)
plt.title("alpha effect",fontsize=20)
ax.set_xlim(0, np.pi/2)
ax.set_ylim(0,1.0)
plt.savefig("P_alpha.png")
plt.clf()
plt.close('all')