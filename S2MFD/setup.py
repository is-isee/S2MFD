from S2MFD.tools import drr2, dth2
from S2MFD.npz_io import NpzIO
import warnings

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
         self.build_turbulent_transport(cfg, grid)
         self.build_lambda(cfg, grid)

   def build_turbulent_transport(self, cfg, grid):
      """乱流粘性 ν_t(r) と乱流熱伝導 κ_t(r) を構築する (Rempel 2005 式 27-30)。

      .. math::
         \\nu_\\Lambda &= \\frac{\\nu_0}{2}
            \\left[1+\\tanh\\frac{r-r_{\\rm tran}+\\Delta}{d_{\\kappa\\nu}}\\right]
            f_c(r) \\\\
         f_c(r) &= \\frac12\\left[1+\\tanh\\frac{r-r_{\\rm bc}}{d_{\\rm bc}}\\right],
         \\quad
         \\Delta = d_{\\kappa\\nu}\\,{\\rm artanh}(2\\alpha_{\\kappa\\nu}-1)

      :math:`\\Delta` は「:math:`r=r_{\\rm tran}` で
      :math:`\\nu_\\Lambda=\\alpha_{\\kappa\\nu}\\nu_0` になる」ようにずらす量。

      拡散項用とΛ効果用で粘性が違う
      ------------------------------
      Rempel 2005 §2.4 は、**Λ効果による角運動量輸送**には上の
      プロファイルをそのまま使い、**レイノルズ応力の拡散項**に使う粘性には
      対流層値の 2%(熱伝導は 0.2%)の下限を張る、としている。

      理由は下部境界の :math:`\\Omega_1=0` 条件との間にせん断層
      (タコクライン)を現実的な時間で形成させるため。放射層で粘性が
      ゼロまで落ちると、そこに角運動量を運べず定常解に到達しない。

      なお「2% にする」は原文では放射層に限定した書き方になっていないが、
      対流層全体で 2% にすると論文表 1 の
      :math:`Q_\\nu^\\Omega/Q_\\Lambda=0.574`
      (粘性散逸がΛ効果入力の 57%)と桁が合わなくなるため、下限として
      解釈している。
      """
      rr = grid.rr
      dkn = cfg.d_kn
      # r = r_tran で alpha_kn * nu0 になるようにずらす
      shift = dkn*np.arctanh(2.0*cfg.alpha_kn - 1.0)
      fc = 0.5*(1.0 + np.tanh((rr - cfg.rr_bc)/cfg.d_bc))
      shape = 0.5*(1.0 + np.tanh((rr - cfg.r_tran + shift)/dkn))*fc

      # 数値安定性のための増幅係数。既定は 1.0 (論文どおり)。
      # 本実装は SSP-RK2 + 中心差分で数値散逸を持たないため、Rempel の
      # MacCormack (交互風上/風下) が暗黙に持つ散逸を補う必要がある場合に
      # 使う。論文からの逸脱なので、1.0 でない値を使ったら必ず記録すること。
      # Λ効果用の粘性には掛けない (駆動の強さを変えないため)。
      fnu = getattr(cfg, 'nu_numerical_factor', 1.0)
      fkp = getattr(cfg, 'kappa_numerical_factor', 1.0)

      self.nu_lam = cfg.nu0*shape                          # Λ効果用
      self.nu_dif = fnu*np.maximum(cfg.nu0*shape,
                                   cfg.nu_floor_frac*cfg.nu0)  # 拡散項用
      self.kappa_t = fkp*np.maximum(cfg.kappa0*shape,
                                    cfg.kappa_floor_frac*cfg.kappa0)

      # r 面上の値 (面 i はセル i-1 と i の境界)。拡散フラックスの係数として
      # 隣接2セルで同一の値を使うために必要。
      for name in ('nu_lam', 'nu_dif', 'kappa_t'):
         arr = getattr(self, name)
         face = np.zeros_like(arr)
         face[1:] = 0.5*(arr[1:] + arr[:-1])
         setattr(self, name + '_m', face)

   def _r_max(self, cfg, grid):
      """Rempel の :math:`r_{\\max}` (Λ 効果と α 効果の基準半径) を返す。

      論文では計算領域の上端そのもの (:math:`0.985R_\\odot`) だが、本実装は
      数値安定性のために領域を切り詰めて走らせることがある。そのとき
      ``grid.rrmax`` をそのまま使うと**駆動の位置まで一緒に動いてしまう**:

      * Λ 効果 (2005 式 33): :math:`\\tanh((r_{\\max}-r)/d)` の遷移層
      * α 効果 (2006 式 17): :math:`\\max[0,1-(r-r_{\\max})^2/d_\\alpha^2]`
        は湧き出しを :math:`r>r_{\\max}-d_\\alpha` に閉じ込める
        (論文 §2.2 は "confines the poloidal source term above
        :math:`r=0.935R_\\odot`" と明記)

      ``cfg.r_max`` を明示すれば領域と切り離せる。指定がなければ従来どおり
      ``grid.rrmax`` に追随する。

      **警告について**: Rempel (2005) §2.5 は
      "we require a vanishing angular momentum flux at the top boundary"
      と述べており、:math:`\\tanh((r_{\\max}-r)/d)` はそのための遷移層である。
      :math:`r_{\\max}` を領域上端からずらすと Λ フラックスが上部境界で
      ゼロにならず、stress-free 条件と整合しなくなる。したがって
      「領域を切り詰めたまま駆動だけ論文の位置に置く」ことは近似ですらない。
      両者がずれている場合は警告を出す。
      """
      r_max = getattr(cfg, 'r_max', None)
      if r_max is None:
         return grid.rrmax
      if abs(r_max - grid.rrmax) > 1e-6*cfg.RSUN:
         warnings.warn(
            f"cfg.r_max ({r_max/cfg.RSUN:.4f} RSUN) が計算領域の上端 "
            f"({grid.rrmax/cfg.RSUN:.4f} RSUN) と違います。Λ 効果の遷移層 "
            f"tanh((r_max-r)/d) は上部境界で角運動量フラックスを消すための"
            f"もの (Rempel 2005 §2.5) なので、ずれていると上部境界を通って"
            f"角運動量が出入りします。論文を再現するなら領域上端を "
            f"0.985 RSUN にしてください。",
            UserWarning, stacklevel=3)
      return r_max

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

      ff = grid.sinTH**n*grid.cosTH*np.tanh((self._r_max(cfg, grid) - grid.RR)/dl)
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
      self.omrr = drr2(self.om, grid.drr2)
      self.omth = dth2(self.om, grid.dth)/grid.RR

   def build_diffusivity(self, cfg, grid):
      """磁気拡散プロファイル (et, etrr) を構築する。"""
      # diffusivity
      if cfg.diffusive_type == 'J08':
         self.et = cfg.etc + 0.5*(cfg.ett - cfg.etc)*(1 + erf((grid.RR-cfg.rrc)/cfg.d))
      elif cfg.diffusive_type == 'H10':
         self.et = cfg.etc + 0.5*cfg.ett*(1 + erf((grid.RR-cfg.rrc)/cfg.dh1)) + 0.5*cfg.ets*(1 + erf((grid.RR-cfg.r1)/cfg.dh2))
      elif cfg.diffusive_type == 'R06':
         # Rempel (2006) 式 (13)-(15)
         #   eta_t = eta_c + f_c(r)[eta_bc - eta_c + f_cz(r)(eta_cz - eta_bc)]
         fc = 0.5*(1 + np.tanh((grid.RR - cfg.rr_bc)/cfg.d_bc))
         fcz = 0.5*(1 + np.tanh((grid.RR - cfg.r_cz)/cfg.d_cz))
         self.et = cfg.eta_c + fc*(cfg.eta_bc - cfg.eta_c
                                   + fcz*(cfg.eta_cz - cfg.eta_bc))
      else:
         raise ValueError(f"unknown diffusive_type: {cfg.diffusive_type!r}")
      self.etrr = drr2(self.et, grid.drr2)

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
      elif cfg.alpha_type == 'R06':
         # Rempel (2006) 式 (16)-(19)。B_phi の動径平均に比例する非局所ソース
         #   S = alpha0 * Bbar_phi(theta) * f_alpha(r) * g_alpha(theta)
         # ここでは r, theta 依存の形状 f_alpha*g_alpha だけを so に入れ、
         # Bbar_phi(theta) との積は時間積分カーネル側で取る
         # (so に B_phi を含められないため)。
         fal = np.maximum(0.0, 1.0 - (grid.RR - self._r_max(cfg, grid))**2
                          / cfg.d_alpha**2)
         gnum = grid.sinTH**2*grid.cosTH
         i0, i1 = grid.margin, grid.ixg - grid.margin
         j0, j1 = grid.margin, grid.jxg - grid.margin
         self.so = cfg.alpha0*fal*gnum/np.abs(gnum[i0:i1, j0:j1]).max()

         # B_phi を平均する放物線カーネル h(r): r_h_bot と r_h_top でゼロ、
         # 中間でピーク。int h(r) dr = 1 に規格化する。
         hker = np.maximum(0.0, (grid.rr - cfg.r_h_bot)*(cfg.r_h_top - grid.rr))
         hker[:i0] = 0.0
         hker[i1:] = 0.0
         norm = (hker*grid.drr).sum()
         self.alpha_kernel = hker/norm if norm > 0 else hker
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