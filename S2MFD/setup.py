from S2MFD.tools import drr2, dth2
from S2MFD.npz_io import NpzIO
import numpy as np
from scipy.special import erf


class Setup(NpzIO):
   """
   Class to configure and initialize physical properties used in simulations.

   Attributes
   ----------
   urr : numpy.ndarray
      Radial component of the meridional flow velocity.
   uth : numpy.ndarray
      Latitudinal component of the meridional flow velocity.
   om : numpy.ndarray
      Angular velocity distribution based on the differential rotation profile.
   omrr : numpy.ndarray
      Radial derivative of angular velocity.
   omth : numpy.ndarray
      Latitudinal derivative of angular velocity.
   et : numpy.ndarray
      Magnetic diffusivity profile.
   etrr : numpy.ndarray
      Radial derivative of the magnetic diffusivity.
   so : numpy.ndarray
      Alpha effect (poloidal source) amplitude profile.
   ibase : int
      Index corresponding to the tachocline region in the radial direction.

   """

   def __init__(self,cfg,grid):
      """
      Initialize the setup object.
      
      Parameters
      ----------
      cfg : S2MFD.Cfg
         Configuration object.
         
      grid : S2MFD.Grid
         Grid object.
      """
      self.build_rotation(cfg, grid)
      self.build_diffusivity(cfg, grid)

      #タコクラインのindex
      self.ibase = np.argmin(abs(grid.rr - cfg.rrc))

      self.build_alpha(cfg, grid)
      self.build_flow(cfg, grid)

      if getattr(cfg, 'dynamics', 'kinematic') != 'kinematic':
         self.build_lambda(cfg, grid)

   def build_lambda(self, cfg, grid):
      """Λ効果 (非等方レイノルズ応力) のプロファイルを構築する。

      Rempel (2005) 式 (14)-(16)。角運動量フラックスを回転軸方向から
      角度 ``cfg.lambda_tilt_deg`` だけ傾けた向きに与える:

      .. math::
         \\Lambda_{r\\varphi} = L\\cos(\\theta+\\varepsilon), \\quad
         \\Lambda_{\\theta\\varphi} = -L\\sin(\\theta+\\varepsilon)

      振幅は :math:`L \\propto \\sin^n\\theta\\cos\\theta
      \\tanh((r_{\\max}-r)/d)` で、対流層上端で滑らかにゼロになる。

      全球格子での赤道対称性
      ----------------------
      齋藤 (2024) のコードは北半球 :math:`[0,\\pi/2]` のみを解くので傾き
      :math:`\\varepsilon` の符号を考える必要がない。S2MFD は全球
      :math:`[0,\\pi]` を解くため、そのまま使うと赤道対称性が壊れる。

      :math:`q_L` が偶関数であるためには :math:`F^r` が偶、
      :math:`F^\\theta` が奇でなければならず、幾何因子 :math:`\\sin^2\\theta`
      が偶であることから :math:`\\Lambda_{r\\varphi}` は偶、
      :math:`\\Lambda_{\\theta\\varphi}` は奇である必要がある。振幅 L は
      :math:`\\cos\\theta` に比例して奇なので、傾き角を
      :math:`\\varepsilon\\,\\mathrm{sgn}(\\cos\\theta)` として南半球で符号を
      反転させるとこの条件を満たす (物理的にも、傾きは両半球で鏡像になる)。
      """
      n = getattr(cfg, 'lambda_n', 2.0)
      dl = getattr(cfg, 'lambda_d', 0.025*cfg.RSUN)
      lam0 = getattr(cfg, 'lambda0', 0.8)
      eps = np.deg2rad(getattr(cfg, 'lambda_tilt_deg', 15.0))

      ff = grid.sinTH**n*grid.cosTH*np.tanh((grid.rrmax - grid.RR)/dl)
      # 規格化は物理セルの最大値で取る (ゴーストセルは外挿なので除く)
      i0, i1 = grid.margin, grid.ixg - grid.margin
      j0, j1 = grid.margin, grid.jxg - grid.margin
      amp = ff[i0:i1, j0:j1].max()
      L = lam0*cfg.om0*ff/amp

      # 傾きの符号を半球ごとに反転させる (赤道対称性の保持)
      sgn = np.sign(grid.cosTH)
      ce, se = np.cos(eps), np.sin(eps)
      self.lam_rp = L*(grid.cosTH*ce - sgn*grid.sinTH*se)
      self.lam_tp = -L*(grid.sinTH*ce + sgn*grid.cosTH*se)

   def build_rotation(self, cfg, grid):
      """差動回転プロファイル (om, omrr, omth) を構築する。"""
      # differential rotation
      if cfg.differential_type == 'J08':
         self.om = cfg.omc + 0.5*(1 + erf((grid.RR-cfg.rrc)/cfg.d))*(cfg.ome - cfg.omc - cfg.c2*grid.cosTH**2)
      elif cfg.differential_type == 'H10':
         self.om = cfg.omc + 0.5*(1 + erf(2*(grid.RR-cfg.rrc)/cfg.dh1))*(cfg.ome + cfg.a2*grid.cosTH**2 + cfg.a4*grid.cosTH**4 - cfg.omc)
      else:
         raise ValueError(f"unknown differential_type: {cfg.differential_type!r}")
      self.omrr = drr2(self.om, grid.drr)
      self.omth = dth2(self.om, grid.dth)/grid.RR

   def build_diffusivity(self, cfg, grid):
      """磁気拡散プロファイル (et, etrr) を構築する。"""
      # diffusivity
      if cfg.diffusive_type == 'J08':
         self.et = cfg.etc + 0.5*(cfg.ett - cfg.etc)*(1 + erf((grid.RR-cfg.rrc)/cfg.d))
      elif cfg.diffusive_type == 'H10':
         self.et = cfg.etc + 0.5*cfg.ett*(1 + erf((grid.RR-cfg.rrc)/cfg.dh1)) + 0.5*cfg.ets*(1 + erf((grid.RR-cfg.r1)/cfg.dh2))
      else:
         raise ValueError(f"unknown diffusive_type: {cfg.diffusive_type!r}")
      self.etrr = drr2(self.et, grid.drr)

   def build_alpha(self, cfg, grid):
      """α効果 (ポロイダル場ソース) プロファイル so を構築する。

      cfg.so0 を変更した後にこのメソッドを呼ぶと so だけを更新できる
      (時間依存パラメタの差分更新用)。
      """
      # alpha effect
      if cfg.alpha_type == 'BL':
         self.so = cfg.so0*0.5 \
            *(1 + erf((grid.RR-cfg.r1)/cfg.d1))*(1 - erf((grid.RR-cfg.RSUN)/cfg.d1)) \
               *grid.cosTH*grid.sinTH
      elif cfg.alpha_type == 'normal':
         self.so = cfg.so0*3*np.sqrt(3)/4 \
            *(1 + erf((grid.RR-cfg.rrc)/cfg.d)) \
               *grid.sinTH**2*grid.cosTH
      elif cfg.alpha_type == 'H10':
         self.so = cfg.so1*0.25 \
            *(1+erf((grid.RR-cfg.r4)/cfg.dh4))*(1-erf((grid.RR-cfg.r5)/cfg.dh5)) \
               *grid.cosTH*grid.sinTH*(1/(1+np.e**(-cfg.gam*(grid.TH[1,:]-np.pi*0.25)))+1/(1+np.e**(-cfg.gam*(-grid.TH[1,:]+np.pi*0.75)))-1)
      else:
         raise ValueError(f"unknown alpha_type: {cfg.alpha_type!r}")

   def build_flow(self, cfg, grid):
      """子午面流プロファイル (urr, uth) を構築する。

      cfg.uu0 を変更した後にこのメソッドを呼ぶと urr/uth だけを更新できる
      (時間依存パラメタの差分更新用)。
      """
      # Meridional flow
      # Meridional flow (Jouve+2008 Model)
      if cfg.meridional_circulation_type == 'J08':
         self.urr = -cfg.uu0*2*(cfg.RSUN - cfg.rrb)/np.pi/grid.RR \
            *(grid.RR-cfg.rrb)**2/(cfg.RSUN - cfg.rrb)**2 \
            *np.sin(np.pi*(grid.RR-cfg.rrb)/(cfg.RSUN-cfg.rrb))*(3*grid.cosTH**2 - 1)
            
         self.uth = cfg.uu0*((3*grid.RR-cfg.rrb)/(cfg.RSUN-cfg.rrb) \
               *np.sin(np.pi*(grid.RR-cfg.rrb)/(cfg.RSUN-cfg.rrb)) \
               + grid.RR*np.pi/(cfg.RSUN-cfg.rrb)*(grid.RR-cfg.rrb)/(cfg.RSUN-cfg.rrb) \
                  *np.cos(np.pi*(grid.RR-cfg.rrb)/(cfg.RSUN-cfg.rrb))) \
               *2*(cfg.RSUN-cfg.rrb)/np.pi/grid.RR*(grid.RR-cfg.rrb)/(cfg.RSUN-cfg.rrb) \
                  *grid.cosTH*grid.sinTH
      # Meridional flow (Dikpati+1999 Model)
      elif cfg.meridional_circulation_type == 'D99':
         xi  = cfg.RSUN/grid.RR  - 1
         xi[grid.RR > cfg.RSUN] = 0 
         
         self.urr = cfg.uu0*(cfg.RSUN/grid.RR) \
            *(-1/(cfg.m+1) + cfg.c1d/(2*cfg.m + 1)*xi**cfg.m - cfg.c2d/(2*cfg.m+cfg.p+1)*xi**(cfg.m+cfg.p)) \
            *xi*grid.sinTH**cfg.q*( (cfg.q+2)*grid.cosTH**2 - grid.sinTH**2)

         self.uth = cfg.uu0*((cfg.RSUN/grid.RR)**3) \
            *(-1+cfg.c1d*xi**cfg.m - cfg.c2d*xi**(cfg.m+cfg.p)) \
            *grid.sinTH**(cfg.q+1)*grid.cosTH
      
      elif cfg.meridional_circulation_type == 'H10':
         xi  = cfg.RSUN/grid.RR  - 1
         xi[grid.RR > cfg.RSUN] = 0
         
         self.urr = (cfg.uu0/cfg.f)*(cfg.RSUN/grid.RR)**2 \
            *(-1/(cfg.m+1) + cfg.c1d/(2*cfg.m + 1)*xi**cfg.m - cfg.c2d/(2*cfg.m+cfg.p+1)*xi**(cfg.m+cfg.p)) \
            *xi*grid.sinTH**cfg.q*((cfg.q+2)*grid.cosTH**2 - grid.sinTH**2)

         self.uth = (cfg.uu0/cfg.f)*((cfg.RSUN/grid.RR)**3) \
            *(-1+cfg.c1d*xi**cfg.m - cfg.c2d*xi**(cfg.m+cfg.p)) \
            *grid.sinTH**(cfg.q+1)*grid.cosTH
      else:
         raise ValueError(
            f"unknown meridional_circulation_type: {cfg.meridional_circulation_type!r}")


      self.urr[grid.RR < cfg.rrb] = 0
      self.uth[grid.RR < cfg.rrb] = 0
      # 境界の外(ゴーストセル)で子午面流の設定
      # 動径境界: urr 反対称 (境界で urr=0)
      for i in range(0,grid.margin):
         self.urr[i           ,:] = - self.urr[2*grid.margin-i-1       ,:] # bottom
         self.urr[grid.ixg-i-1,:] = - self.urr[grid.ixg-2*grid.margin+i,:] # top

      # 緯度境界 (θ=0, π 回転軸): uth 反対称
      for j in range(0,grid.margin):
         self.uth[:,j           ] = - self.uth[:,2*grid.margin - j - 1     ] # north pole
         self.uth[:,grid.jxg-j-1] = - self.uth[:,grid.jxg-2*grid.margin + j] # south pole