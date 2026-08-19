"""背景場プロファイル (差動回転・子午面流・拡散・α効果) の確認スクリプト。

使い方:
    python check_profile.py [parameter_file]   # 既定は parameters/hotta10.py
"""
import sys

import matplotlib.pyplot as plt
import numpy as np

import S2MFD

# 確認したいパターンを指定 (コマンドライン第1引数でも指定可)
parameter_file = sys.argv[1] if len(sys.argv) > 1 else 'parameters/hotta10.py'
cfg = S2MFD.Cfg(parameter_file)
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
plt.xlabel('length(cm)')
plt.ylabel('length(cm)')
plt.title("Meridional Flow")
plt.ylim(-7e10,7e10)
plt.xlim(0,7e10)
plt.show()

# Diffusivity
plt.figure(figsize=(6, 6))  # 描画領域を正方形にする
plt.xlabel('$r/R$')
plt.ylabel(r'$\eta{(cm^2s^{-1})}$')
plt.ylim(1e8,1e13)
plt.xlim(np.min(grid.RR)/cfg.RSUN,np.max(grid.RR)/cfg.RSUN)
plt.plot(grid.RR[:,1]/cfg.RSUN,setup.et[:,1])
plt.yscale('log')
plt.title("Diffusivity")
plt.show()

# 格子数に依存しないよう θ=60°, r=0.975R のインデックスを計算で求める
jth60 = np.argmin(abs(grid.th - 60/180*np.pi))
ir0975 = np.argmin(abs(grid.rr - 0.975*cfg.RSUN))

# alpha (θ=60°)
plt.figure(figsize=(6, 6))
plt.plot(grid.RR[:, jth60]/cfg.RSUN, setup.so[:, jth60])
plt.xlabel('$r/R$')
plt.ylabel(r'$\alpha{(cm s^{-1})}$')
plt.xlim(np.min(grid.RR)/cfg.RSUN,np.max(grid.RR)/cfg.RSUN)
plt.title(r"alpha effect ($\theta=60^\circ$)")
plt.show()

# alpha (r=0.975R)
plt.figure(figsize=(6, 6))
plt.plot(grid.th*180/np.pi, setup.so[ir0975, :])
plt.xlabel(r'$\theta$')
plt.ylabel(r'$\alpha{(cm s^{-1})}$')
plt.xlim(0,np.max(grid.th*180/np.pi))
plt.title("alpha effect (r=0.975R)")
plt.show()
