sys.path.append('../')
import S2MFD

# 確認したいパターンを自分で指定
cfg = S2MFD.Cfg('parameters/hotta10.py')
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
plt.show()

# Meridional
U_x = setup.urr * grid.cosTH - setup.uth * grid.sinTH
U_y = setup.urr * grid.sinTH + setup.uth * grid.cosTH
fig, ax = plt.subplots()
plt.quiver(grid.Y, grid.X, U_y, U_x)
plt.title('The Flow Fields')
ax.set_aspect('equal')
plt.xlabel('radius')
plt.ylim(-7e10,7e10)
plt.xlim(0,7e10)
plt.show()

# Diffusivity
plt.figure(figsize=(6, 6))  # 描画領域を正方形にする
plt.plot(setup.et[:,43])
ax.set_aspect('equal')
plt.yscale('log')
ax.set_box_aspect(1)  # 縦横比を1:1に設定 (Matplotlib v3.3+)
plt.ylim(1e8,1e13)
plt.xlim(0,cfg.ix)
plt.show()

# alpha (θ＝60)
plt.figure(figsize=(6, 6))  # 描画領域を正方形にする
plt.plot(setup.so[:,43])
ax.set_aspect('equal')
ax.set_box_aspect(1)  # 縦横比を1:1に設定 (Matplotlib v3.3+)
plt.xlim(0,cfg.ix)
plt.show()
# alpha (r=0.975R)
plt.figure(figsize=(6, 6))  # 描画領域を正方形にする
plt.plot(setup.so[127,:])
ax.set_aspect('equal')
ax.set_box_aspect(1)  # 縦横比を1:1に設定 (Matplotlib v3.3+)
plt.xlim(0,cfg.jx)
plt.show()