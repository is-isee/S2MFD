from dataclasses import dataclass, field
import numpy as np
from scipy.special import erf

_SQRT_PI_2 = 0.5*np.sqrt(np.pi)

from S2MFD.npz_io import NpzIO


@dataclass
class Grid(NpzIO):
   """
   Class for managing the grid data.

   Attributes
   ----------
   ix : int
      Number of grid points in the radial direction.
   jx : int
      Number of grid points in the colatitudinal direction.
   margin : int
      Number of margin points on each side of the grid.
   ixg : int
      ix + 2*margin.
   jxg : int
      jx + 2*margin.
   rrmin : float
      Minimum value for the radial direction.
   rrmax : float
      Maximum value for the radial direction.
   thmin : float
      Minimum value for the colatitudinal direction.
   thmax : float
      Maximum value for the colatitudial direction.
   drr : numpy.ndarray
      Radial cell width ``rrm[i+1] - rrm[i]`` (ixg,). 非一様格子では
      i に依存する。``stretch = 0`` なら全要素が等しい。
   drrm : numpy.ndarray
      隣り合うセル中心の距離 ``rr[i] - rr[i-1]`` (ixg,)。**面での勾配**は
      これで割る。一様格子では ``drr`` と一致するが、非一様格子では
      別物なので混同しないこと。
   drr2 : numpy.ndarray
      ``rr[i+1] - rr[i-1]`` (ixg,)。セル中心での 2 セル幅の中心差分用。
   dth : float
      Colatitudinal grid spacing (緯度方向は一様のまま)。
   stretch : float
      動径方向の格子集中度。0 で一様。>0 で ``stretch_center`` 付近に
      点を集める (点の密度が ``1 + stretch*exp(-((r-rc)/w)^2)`` に比例)。
   stretch_center, stretch_width : float or sequence
      集中させる半径とその幅 [cm]。``stretch_width = 0`` で自動
      (領域幅の 15%)。配列にすると複数箇所に同時に集中できる
      (例: タコクラインと表面)。
   rr : numpy.ndarray
      Array of radial grid points.
   th : numpy.ndarray
      Array of colatitudinal grid points.
   RR : numpy.ndarray: 
      Radial coordinate np.meshgrid array.
   TH : numpy.ndarray
      Colatitudinal coordinate np.meshgrid array.
   RRm : numpy.ndarray
      Radial coordinate meshgrid array for centered points.
   THm : numpy.ndarray
      Colatitudinal coordinate meshgrid array for centered points.
   sinTH : numpy.ndarray
      numpy.sin(TH)
   cosTH : numpy.ndarray
      numpy.cos(TH)
   sinTHm : numpy.ndarray
      numpy.sin(THm) (face-centered)
   X : numpy.ndarray
      Cartesian x-coordinates based on radial and colatitudinal grids.
   Y : numpy.ndarray
      Cartesian y-coordinates based on radial and colatitudinal grids.
   """
   ix: int
   jx: int
   ixg: int = field(init=False)
   jxg: int = field(init=False)
   margin: int
   rrmin: float
   rrmax: float
   thmin: float
   thmax: float
   stretch: object = 0.0
   stretch_center: object = 0.0
   stretch_width: object = 0.0
   drr: np.ndarray = field(init=False)
   drrm: np.ndarray = field(init=False)
   drr2: np.ndarray = field(init=False)
   dth: float = field(init=False)
   rr: np.ndarray = field(init=False)
   rrm: np.ndarray = field(init=False)
   wfm: np.ndarray = field(init=False)
   th: np.ndarray = field(init=False)
   
   RR: np.ndarray = field(init=False)
   TH: np.ndarray = field(init=False)
   RRm: np.ndarray = field(init=False) 
   THm: np.ndarray = field(init=False)
   sinTH: np.ndarray = field(init=False)
   cosTH: np.ndarray = field(init=False)
   sinTHm: np.ndarray = field(init=False)
   X: np.ndarray = field(init=False)
   Y: np.ndarray = field(init=False)
   
   @classmethod
   def from_cfg(cls, cfg):
      """``Cfg`` から格子を作る.

      設定ファイルに格子の指定がどう書かれていても、格子の作り方が
      **1 箇所に決まる**ようにするための入口。伸縮格子の既定値
      (``grid_stretch`` など) はここでだけ解釈する。

      Examples
      --------
      >>> import S2MFD
      >>> cfg = S2MFD.build_cfg('parameters/rempel06_paper.py')
      >>> grid = S2MFD.Grid.from_cfg(cfg)
      >>> grid.ix, grid.jx
      (108, 72)
      """
      return cls(
         ix=cfg.ix, jx=cfg.jx, margin=cfg.margin,
         rrmin=cfg.rrmin, rrmax=cfg.rrmax,
         thmin=cfg.thmin, thmax=cfg.thmax,
         stretch=getattr(cfg, 'grid_stretch', 0.0),
         stretch_center=getattr(cfg, 'grid_stretch_center',
                                0.5*(cfg.rrmin + cfg.rrmax)),
         stretch_width=getattr(cfg, 'grid_stretch_width', 0.0),
      )

   def _face_positions(self):
      """セル境界 (面) の半径を ixg+1 個返す。

      点の密度を **物理半径 :math:`r` の関数**として与え、その積分
      :math:`k(r)` を Newton 法で逆に引く。

      .. math::
         \\rho(r) = 1 + \\sum_i a_i
             \\exp\\!\\left[-\\left(\\frac{r-r_i}{w_i}\\right)^2\\right],
         \\qquad
         k(r) = C\\!\\int_{r_{\\min}}^{r}\\!\\rho(r')\\,dr'

      積分は誤差関数で**解析的に**書ける:

      .. math::
         \\int \\exp\\!\\left[-\\left(\\frac{r-r_i}{w_i}\\right)^2\\right] dr
         = \\frac{w_i\\sqrt{\\pi}}{2}\\,{\\rm erf}\\!
           \\left(\\frac{r-r_i}{w_i}\\right)

      ので :math:`k(r)` は :math:`C^\\infty`。逆写像は Newton 法
      (:math:`k'=C\\rho` も解析的) で機械精度まで収束させるため、
      **格子生成に補間を一切使わない**。

      ``stretch`` / ``stretch_center`` / ``stretch_width`` はスカラーでも
      配列でもよい。配列にすると複数の領域 (タコクラインと表面など) に
      同時に点を集められる。

      なぜ解析的な写像でなければならないか
      ------------------------------------
      当初は「点の密度を数値積分して逆に引く」方式にしていたが、
      区分線形補間のせいで格子間隔関数が :math:`C^0` にしかならず、
      **その折れ目が 1 次の誤差を注入して全体が 1 次精度に落ちた**
      (``scratchpad/order_test.py`` で実測: 面平均の収束次数が 0.9)。
      :math:`C^\\infty` の写像に変えるだけで 2 次に戻る
      (``order2.py``: 単純平均 2.00、拡散 2.00、解の誤差 2.01)。
      格子生成器の滑らかさが離散化の次数を決める、という教訓。

      ``stretch == 0`` のときは一様格子の式をそのまま返す。指数関数を
      経由すると丸めで最終桁が動き、既存結果との**ビット一致**が
      壊れるためで、ここは意図的に分岐している。
      """
      dxi = (self.rrmax - self.rrmin)/self.ix
      k = np.arange(self.ixg + 1) - self.margin
      amp = np.atleast_1d(np.asarray(self.stretch, dtype=float))
      if not np.any(amp != 0.0):
         return self.rrmin + dxi*k

      cen = np.atleast_1d(np.asarray(self.stretch_center, dtype=float))
      wid = np.atleast_1d(np.asarray(self.stretch_width, dtype=float)).copy()
      # 幅 0 は「自動」= 領域幅の 15%
      wid = np.where(wid <= 0.0, 0.15*(self.rrmax - self.rrmin), wid)
      cen = np.broadcast_to(cen, amp.shape)
      wid = np.broadcast_to(wid, amp.shape)

      def density(r):
         out = np.ones_like(np.asarray(r, dtype=float))
         for a, c, w in zip(amp, cen, wid):
            out = out + a*np.exp(-((r - c)/w)**2)
         return out

      def kk(r):
         out = np.asarray(r, dtype=float) - self.rrmin
         for a, c, w in zip(amp, cen, wid):
            out = out + a*w*_SQRT_PI_2*(erf((r - c)/w)
                                        - erf((self.rrmin - c)/w))
         return out

      norm = self.ix/kk(self.rrmax)
      target = k/norm                      # k(r) = target を解く
      # Newton 法。k は単調増加 (density > 0) なので必ず収束する。
      r = self.rrmin + dxi*k               # 一様格子を初期推定に使う
      for _ in range(60):
         dr = (kk(r) - target)/density(r)
         r = r - dr
         if np.max(np.abs(dr)) < 1.0e-12*(self.rrmax - self.rrmin):
            break
      # ゴーストセルは端の間隔で線形に延長する (領域外まで密度関数を
      # 使うと、集中点の裾で間隔が不自然に変わることがある)
      h0 = 1.0/(norm*density(self.rrmin))
      h1 = 1.0/(norm*density(self.rrmax))
      r = np.where(k < 0.0, self.rrmin + h0*k, r)
      r = np.where(k > self.ix, self.rrmax + h1*(k - self.ix), r)
      return r

   def __post_init__(self):
      # dr,dθの設定
      self.ixg = self.ix + 2*self.margin
      self.jxg = self.jx + 2*self.margin
      self.dth = (self.thmax - self.thmin)/self.jx

      # --- 動径方向の格子 -----------------------------------------------
      # 有限体積なので **面が主** で、セル中心は面の中点にする。こうすると
      # sum(q*drr) が厳密に体積積分になり、発散のテレスコープがそのまま
      # 成り立つ (非一様でも保存は機械精度)。
      #
      # ゴーストセルも含めて ixg+1 枚の面を作る。面の位置は伸縮写像
      #   xi -> R(xi),   R' = 1/(rho(xi) * I)
      # で決める。rho は点の密度で、stretch = 0 なら rho = 1 となり
      # **一様格子と厳密に一致する** (回帰の安全弁)。
      self.rrm = self._face_positions()
      if not np.any(np.atleast_1d(self.stretch) != 0.0):
         # 一様格子は従来の式をそのまま使う。面の差を取ると丸めで最終桁が
         # 動き、既存結果とのビット一致が壊れるため。
         dxi = (self.rrmax - self.rrmin)/self.ix
         rr0 = self.rrmin + dxi*(0.5 - self.margin)
         self.rr = rr0 + dxi*np.arange(self.ixg)
         self.drr = np.full(self.ixg, dxi)
         self.drrm = np.full(self.ixg, dxi)
         self.drr2 = np.full(self.ixg, 2.0*dxi)
      else:
         self.rr = 0.5*(self.rrm[:-1] + self.rrm[1:])
         self.drr = self.rrm[1:] - self.rrm[:-1]
         self.drrm = np.empty(self.ixg)
         self.drrm[1:] = self.rr[1:] - self.rr[:-1]
         self.drrm[0] = self.drrm[1]
         self.drr2 = np.empty(self.ixg)
         self.drr2[1:-1] = self.rr[2:] - self.rr[:-2]
         self.drr2[0] = 2.0*self.drrm[1]
         self.drr2[-1] = 2.0*self.drrm[-1]

      # 面への線形補間の重み: q_face[i] = wm[i]*q[i-1] + (1-wm[i])*q[i]
      # 一様格子では厳密に 0.5 になる (ビット一致の保証)。
      self.wfm = np.full(self.ixg, 0.5)
      if np.any(np.atleast_1d(self.stretch) != 0.0):
         self.wfm[1:] = (self.rr[1:] - self.rrm[1:self.ixg])/self.drrm[1:]

      #座標thの設定
      th0 = self.thmin + self.dth*(0.5 - self.margin)
      self.th = th0 + self.dth*np.arange(self.jxg)
         
      self.RR ,self.TH  = np.meshgrid(self.rr, self.th,indexing='ij')
      self.RRm = np.zeros_like(self.RR)
      self.THm = np.zeros_like(self.TH)
      
      # 面の位置は「セル中心の平均」ではなく **本物の面**を入れる。
      # 一様格子では代数的には両者が一致するので、丸めまで含めて従来と
      # 同じ値になるよう旧式をそのまま使う (ビット一致の保証)。
      if not np.any(np.atleast_1d(self.stretch) != 0.0):
         self.RRm[1:self.ixg,:] = 0.5*(self.RR[1:self.ixg,:]
                                       + self.RR[0:self.ixg-1,:])
      else:
         self.RRm[:, :] = self.rrm[:self.ixg, None]
      self.THm[:,1:self.jxg] = 0.5*(self.TH[:,1:self.jxg] + self.TH[:,0:self.jxg-1])
      
      self.sinTH = np.sin(self.TH)
      self.cosTH = np.cos(self.TH)
      self.sinTHm = np.sin(self.THm)
      
      self.X, self.Y = self.RR * np.cos(self.TH), self.RR * np.sin(self.TH)