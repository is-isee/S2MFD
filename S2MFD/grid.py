from dataclasses import dataclass, field
import numpy as np

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
   stretch_center, stretch_width : float
      集中させる半径とその幅 [cm]。
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
   stretch: float = 0.0
   stretch_center: float = 0.0
   stretch_width: float = 1.0
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
   
   def _face_positions(self):
      """セル境界 (面) の半径を ixg+1 個返す。

      計算座標 :math:`s = k/ix` を一様に取り、**解析的な** sinh 写像で
      物理半径に移す。

      .. math::
         r(s) = r_{\\min} + L\\,
            \\frac{\\sinh[b(s-s_0)] + \\sinh(b s_0)}
                 {\\sinh[b(1-s_0)] + \\sinh(b s_0)}

      :math:`b>0` で :math:`s=s_0` の付近に点が集まる。
      ``stretch`` が :math:`b`、``stretch_center`` が集中させたい半径で、
      :math:`s_0` はそこから決める。

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
      if self.stretch == 0.0:
         return self.rrmin + dxi*k

      L = self.rrmax - self.rrmin
      b = self.stretch
      s0 = (self.stretch_center - self.rrmin)/L
      s = k/self.ix
      den = np.sinh(b*(1.0 - s0)) + np.sinh(b*s0)
      return self.rrmin + L*(np.sinh(b*(s - s0)) + np.sinh(b*s0))/den

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
      if self.stretch == 0.0:
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
      if self.stretch != 0.0:
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
      if self.stretch == 0.0:
         self.RRm[1:self.ixg,:] = 0.5*(self.RR[1:self.ixg,:]
                                       + self.RR[0:self.ixg-1,:])
      else:
         self.RRm[:, :] = self.rrm[:self.ixg, None]
      self.THm[:,1:self.jxg] = 0.5*(self.TH[:,1:self.jxg] + self.TH[:,0:self.jxg-1])
      
      self.sinTH = np.sin(self.TH)
      self.cosTH = np.cos(self.TH)
      self.sinTHm = np.sin(self.THm)
      
      self.X, self.Y = self.RR * np.cos(self.TH), self.RR * np.sin(self.TH)