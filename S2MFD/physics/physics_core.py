
import numpy as np
from S2MFD.tools import drr1, drr2, dth1, dth2
from numba import njit

@njit
def poloidal_mag(Aph, RR, sinTH, drr, dth):
   """
   Calculate the poloidal magnetic field.

   Parameters
   ----------
   Aph : numpy.ndarray
      Longitudinal vector potential
   RR : numpy.ndarray
      Radial coordinate
   sinTH : numpy.ndarray
   drr : numpy.ndarray
      Radial grid spacing
   dth : numpy.ndarray
      Colatitudinal grid spacing

   Returns
   -------
   tuple of numpy.ndarray, float
      - Brr : Radial  magnetic field
      - Bth : Latitudinal magnetic field
         
   """
   Brr = + dth2(sinTH*Aph,dth)/RR/sinTH
   Bth = - drr2(   RR*Aph,drr)/RR
   
   return Brr, Bth

@njit
def advection(Bph, Aph, RR, sinTH, urr,uth,drr,dth):
   """
   Calculate the advection terms of the magnetic field.

   Parameters
   ----------
   Bph : numpy.ndarray
      Longitudinal magnetic field
   Aph : numpy.ndarray
      Longitudinal vector potential
   RR : numpy.ndarray
      Radial coordinate
   sinTH : numpy.ndarray
   urr : numpy.ndarray
      Radial velocity
   uth : numpy.ndarray
      Colatitudinal velocity
   drr : numpy.ndarray
      Radial grid spacing
   dth : numpy.ndarray
      Colatitudinal grid spacing

   Returns
   -------
   tuple of numpy.ndarray
      - Bph_adrr : Radial advection of Bph
      - Bph_adth : Colatitudinal advection of Bph
      - Aph_adrr : Radial advection of Aph
      - Aph_adth : Colatitudinal advection of Aph
   """
   # 磁場の微分(移流量)
   # Bph_adrr = - drr2(Bph/RR   ,drr)*urr*RR #  動径方向の移流(Bph)
   # Bph_adth = - dth2(Bph/sinTH,dth)*uth*sinTH/RR #  緯度方向の移流(Bph)

   Bph_adrr = - drr2(Bph*urr*RR,drr)/RR #  動径方向の移流(Bph)
   Bph_adth = - dth2(Bph*uth   ,dth)/RR #  緯度方向の移流(Bph)
   
   Aph_adrr = - drr2(Aph*RR   ,drr)*urr/RR       # 動径方向の移流(Aph)
   Aph_adth = - dth2(Aph*sinTH,dth)*uth/sinTH/RR # 緯度方向の移流(Aph)

   return Bph_adrr, Bph_adth, Aph_adrr, Aph_adth

@njit
def diffusion(Bph, Aph, RR, sinTH, RRm, sinTHm, drr, dth, et, etrr):
   """
   Calculate the diffusion terms of the magnetic field.

   Parameters
   ----------
   Bph : numpy.ndarray
      Longitudinal magnetic field
   Aph : numpy.ndarray
      Longitudinal vector potential
   RR : numpy.ndarray
      Radial coordinate
   sinTH : numpy.ndarray
   drr : numpy.ndarray
      Radial grid spacing
   dth : numpy.ndarray
      Colatitudinal grid spacing
   et : numpy.ndarray
      magnetic diffusivity
   etrr : numpy.ndarray
      Radial gradient of magnetic diffusivity
   
   Returns
   -------
   tuple of numpy.ndarray
      - Bph_dfrr : Radial diffusion of Bph
      - Bph_dfth : Colatitudinal diffusion of Bph
      - Bph_dfex : Extra diffusion term of Bph
      - Bph_dfrrg : Radial gradient diffusion of Bph
      - Aph_dfrr : Radial diffusion of Aph
      - Aph_dfth : Colatitudinal diffusion of Aph
      - Aph_dfex : Extra diffusion term of Aph
      
   """   
   # magnetic derivative
   Bphrr = drr1(Bph,drr,'up')
   Bphth = dth1(Bph,dth,'up')
   Aphrr = drr1(Aph,drr,'up')
   Aphth = dth1(Aph,dth,'up')
   
   Bph_dfrr = + et*drr1(RRm**2*Bphrr,drr,'dw')/RR**2
   Bph_dfth = + et*dth1(sinTHm*Bphth,dth,'dw')/RR**2/sinTH
   Bph_dfex = - et*Bph/RR**2/sinTH**2
   
   Aph_dfrr = + et*drr1(RRm**2*Aphrr,drr,'dw')/RR**2
   Aph_dfth = + et*dth1(sinTHm*Aphth,dth,'dw')/RR**2/sinTH
   Aph_dfex = - et*Aph/RR**2/sinTH**2
   
   # diffusivity gradient influence
   Bph_dfrrg = etrr*drr2(RR*Bph,drr)/RR
   
   return Bph_dfrr, Bph_dfth, Bph_dfex, Bph_dfrrg, Aph_dfrr, Aph_dfth, Aph_dfex

@njit
def omega_effect(Brr, Bth, RR, sinTH, omrr, omth):
   """
   Calculate the omega effect.

   Parameters
   ----------
   Brr : numpy.ndarray
      Radial magnetic field
   Bth : numpy.ndarray
      Colatitudinal magnetic field
   RR : numpy.ndarray
      Radial coordinate
   sinTH : numpy.ndarray
   omrr : numpy.ndarray
      Radial gradient of angular velocity
   omth : numpy.ndarray
      Colatitudinal gradient of angular velocity

   Returns
   -------
   tuple of numpy.ndarray
   Bph_omrr : Radial omega effect term of Bph
   Bph_omrh : Colatitudinal omega effect term of Bph
   
   """
   Bph_omrr = Brr*omrr*RR*sinTH
   Bph_omth = Bth*omth*RR*sinTH
   
   return Bph_omrr, Bph_omth

def alpha_effect(Bph, Aph, rr, ibase, so, alpha_type):
   """
   Calculate the alpha effect.

   Parameters
   ----------
   Bph : numpy.ndarray
      Longitudinal magnetic field
   Aph : numpy.ndarray
      Longitudinal vector potential
   rr : numpy.ndarray
      Radial coordinate
   ibase : int
      Radial index for the base of the convection zone
   so : numpy.ndarray
      Source term amplitude profile (2D)
   alpha_type : str
      'BL', 'H10' (non-local) or 'normal' (local)
   Returns
   -------
   Aph_sour : numpy.ndarray
      Source term due to alpha effect
   """
   if alpha_type == 'BL' or alpha_type == 'H10':
      # 非局所 (Babcock-Leighton): タコクラインの Bph を全動径に放送
      Bphso = Bph[ibase, :][np.newaxis, :]
      Aph_sour = so*Bphso/(1 + (Bphso)**2)
   elif alpha_type == 'normal':
      Aph_sour = so*Bph/(1 + (Bph)**2)
   else:
      raise ValueError(f"unknown alpha_type: {alpha_type!r}")

   return Aph_sour
ALPHA_CODE = {'BL': 0, 'H10': 0, 'normal': 1}


def _make_time_marching_kernel(fast, nonlocal_alpha, separable=False):
   """time_marching の内部カーネルを生成する。

   fast=True  : 除算を逆数乗算に置換 + numba fastmath (既定)
   fast=False : 参照実装と同じ演算順序 (ビット一致するが約17倍遅い)
   nonlocal_alpha : True なら Babcock-Leighton 型 (BL/H10)、False なら 'normal'
   separable : True なら背景場 (urr, uth, et, etrr, omth, so) を
      rank-1 分解 f(r)g(theta) の 1D 配列から再構成する。
      130x130 の 2D 配列 6 本を読まずに済み、メモリトラフィックが約半分になる。
      GA は 30 プロセス同時実行で帯域律速になるため効果が大きい。

   alpha 種別はコンパイル時に固定する。実行時分岐を内側ループに残すと
   ベクトル化が阻害され 2 倍以上遅くなるため。
   """

   @njit(fastmath=fast, boundscheck=False)
   def kernel(Bph, Aph, dt, rr, sth, rrm, sthm, drr, dth,
              urr, uth, et, etrr, omrr, omth, so, ibase, alpha_code,
              inv_rr, inv_rr2, inv_sth, inv_sth2, alpha_fac, Bphm, Aphm,
              urr_u, urr_v, uth_u, uth_v, et_u, et_v, etrr_u, etrr_v,
              omth_u, omth_v, so_u, so_v):
      ixg, jxg = Bph.shape
      idrr = 1.0/drr
      idth = 1.0/dth

      # --- 内部セル: 分岐なし・除算なし (fast=True 時) ---
      for i in range(1, ixg - 1):
         rr_i = rr[i]
         rr_im = rr[i - 1]
         rr_ip = rr[i + 1]
         rrm_i2 = rrm[i]**2
         rrm_ip2 = rrm[i + 1]**2
         rr2 = rr_i**2
         irr = inv_rr[i]
         irr2 = inv_rr2[i]
         if separable:
            urr_ui = urr_u[i]
            urr_uip = urr_u[i + 1]
            urr_uim = urr_u[i - 1]
            uth_ui = uth_u[i]
            et_ui = et_u[i]
            etrr_ui = etrr_u[i]
            omth_ui = omth_u[i]
            so_ui = so_u[i]
         for j in range(1, jxg - 1):
            sth_j = sth[j]
            sth_jm = sth[j - 1]
            sth_jp = sth[j + 1]
            sthm_j = sthm[j]
            sthm_jp = sthm[j + 1]
            isth = inv_sth[j]
            isth2 = inv_sth2[j]

            bph_c = Bph[i, j]
            aph_c = Aph[i, j]
            bph_im = Bph[i - 1, j]
            bph_ip = Bph[i + 1, j]
            bph_jm = Bph[i, j - 1]
            bph_jp = Bph[i, j + 1]
            aph_im = Aph[i - 1, j]
            aph_ip = Aph[i + 1, j]
            aph_jm = Aph[i, j - 1]
            aph_jp = Aph[i, j + 1]
            if separable:
               et_c = et_ui*et_v[j]
               etrr_c = etrr_ui*etrr_v[j]
               so_c = so_ui*so_v[j]
               omth_c = omth_ui*omth_v[j]
               urr_c = urr_ui*urr_v[j]
               urr_ip_c = urr_uip*urr_v[j]
               urr_im_c = urr_uim*urr_v[j]
               uth_c = uth_ui*uth_v[j]
               uth_jp_c = uth_ui*uth_v[j + 1]
               uth_jm_c = uth_ui*uth_v[j - 1]
            else:
               et_c = et[i, j]
               etrr_c = etrr[i, j]
               so_c = so[i, j]
               omth_c = omth[i, j]
               urr_c = urr[i, j]
               urr_ip_c = urr[i + 1, j]
               urr_im_c = urr[i - 1, j]
               uth_c = uth[i, j]
               uth_jp_c = uth[i, j + 1]
               uth_jm_c = uth[i, j - 1]

            if fast:
               Brr = (sth_jp*aph_jp - sth_jm*aph_jm)*idth*0.5*irr*isth
               Bth = -((rr_ip*aph_ip - rr_im*aph_im)*idrr*0.5*irr)
               Bph_adrr = -((bph_ip*urr_ip_c*rr_ip
                             - bph_im*urr_im_c*rr_im)*idrr*0.5*irr)
               Bph_adth = -((bph_jp*uth_jp_c
                             - bph_jm*uth_jm_c)*idth*0.5*irr)
               Aph_adrr = -((aph_ip*rr_ip - aph_im*rr_im)*idrr*0.5*urr_c*irr)
               Aph_adth = -((aph_jp*sth_jp - aph_jm*sth_jm)
                            * idth*0.5*uth_c*isth*irr)
               div_b_rr = (rrm_ip2*((bph_ip - bph_c)*idrr)
                           - rrm_i2*((bph_c - bph_im)*idrr))*idrr
               div_a_rr = (rrm_ip2*((aph_ip - aph_c)*idrr)
                           - rrm_i2*((aph_c - aph_im)*idrr))*idrr
               div_b_th = (sthm_jp*((bph_jp - bph_c)*idth)
                           - sthm_j*((bph_c - bph_jm)*idth))*idth
               div_a_th = (sthm_jp*((aph_jp - aph_c)*idth)
                           - sthm_j*((aph_c - aph_jm)*idth))*idth
               Bph_dfrr = et_c*div_b_rr*irr2
               Bph_dfth = et_c*div_b_th*irr2*isth
               Bph_dfex = -(et_c*bph_c*irr2*isth2)
               Aph_dfrr = et_c*div_a_rr*irr2
               Aph_dfth = et_c*div_a_th*irr2*isth
               Aph_dfex = -(et_c*aph_c*irr2*isth2)
               Bph_dfrrg = etrr_c*((rr_ip*bph_ip
                                        - rr_im*bph_im)*idrr*0.5)*irr
               if nonlocal_alpha:
                  # 非局所 (BL/H10): b/(1+b^2) は j のみに依存 → 事前計算済み
                  Aph_sour = so_c*alpha_fac[j]
               else:
                  # 局所 ('normal'): セルごとの Bph に依存するため除算が残る
                  Aph_sour = so_c*bph_c/(1 + bph_c**2)
            else:
               Brr = (sth_jp*aph_jp - sth_jm*aph_jm)/dth*0.5/rr_i/sth_j
               Bth = -((rr_ip*aph_ip - rr_im*aph_im)/drr*0.5/rr_i)
               Bph_adrr = -((bph_ip*urr[i+1, j]*rr_ip
                             - bph_im*urr[i-1, j]*rr_im)/drr*0.5/rr_i)
               Bph_adth = -((bph_jp*uth[i, j+1]
                             - bph_jm*uth[i, j-1])/dth*0.5/rr_i)
               Aph_adrr = -((aph_ip*rr_ip - aph_im*rr_im)/drr*0.5*urr_c/rr_i)
               Aph_adth = -((aph_jp*sth_jp - aph_jm*sth_jm)
                            / dth*0.5*uth_c/sth_j/rr_i)
               div_b_rr = (rrm_ip2*((bph_ip - bph_c)/drr)
                           - rrm_i2*((bph_c - bph_im)/drr))/drr
               div_a_rr = (rrm_ip2*((aph_ip - aph_c)/drr)
                           - rrm_i2*((aph_c - aph_im)/drr))/drr
               div_b_th = (sthm_jp*((bph_jp - bph_c)/dth)
                           - sthm_j*((bph_c - bph_jm)/dth))/dth
               div_a_th = (sthm_jp*((aph_jp - aph_c)/dth)
                           - sthm_j*((aph_c - aph_jm)/dth))/dth
               Bph_dfrr = et_c*div_b_rr/rr2
               Bph_dfth = et_c*div_b_th/rr2/sth_j
               Bph_dfex = -(et_c*bph_c/rr2/sth_j**2)
               Aph_dfrr = et_c*div_a_rr/rr2
               Aph_dfth = et_c*div_a_th/rr2/sth_j
               Aph_dfex = -(et_c*aph_c/rr2/sth_j**2)
               Bph_dfrrg = etrr_c*((rr_ip*bph_ip
                                        - rr_im*bph_im)/drr*0.5)/rr_i
               if nonlocal_alpha:
                  b_src = Bph[ibase, j]
               else:
                  b_src = bph_c
               Aph_sour = so_c*b_src/(1 + b_src**2)

            Bph_omrr = Brr*omrr[i, j]*rr_i*sth_j
            Bph_omth = Bth*omth_c*rr_i*sth_j

            dBph = ((Bph_adrr + Bph_adth)
                    + ((Bph_dfrr + Bph_dfth + Bph_dfrrg) + Bph_dfex)
                    + (Bph_omrr + Bph_omth))
            dAph = ((Aph_adrr + Aph_adth)
                    + ((Aph_dfrr + Aph_dfth) + Aph_dfex)
                    + Aph_sour)
            Bphm[i, j] = bph_c + dt*dBph
            Aphm[i, j] = aph_c + dt*dAph

      # --- 境界セル (全体の約3%): 参照実装のゼロ埋め規則を分岐で再現 ---
      n_edge = ixg*2 + (jxg - 2)*2
      for idx in range(n_edge):
         if idx < ixg:
            i = idx
            j = 0
         elif idx < 2*ixg:
            i = idx - ixg
            j = jxg - 1
         elif idx < 2*ixg + (jxg - 2):
            i = 0
            j = idx - 2*ixg + 1
         else:
            i = ixg - 1
            j = idx - 2*ixg - (jxg - 2) + 1

         has_im1 = i >= 1
         has_ip1 = i <= ixg - 2
         has_jm1 = j >= 1
         has_jp1 = j <= jxg - 2
         inner_i = has_im1 and has_ip1
         inner_j = has_jm1 and has_jp1

         rr_c = rr[i]
         sth_c = sth[j]
         bph_c = Bph[i, j]
         aph_c = Aph[i, j]
         rr2 = rr_c**2

         if inner_j:
            d_sth_aph = (sth[j+1]*Aph[i, j+1] - sth[j-1]*Aph[i, j-1])/dth*0.5
            d_buth = (Bph[i, j+1]*uth[i, j+1]
                      - Bph[i, j-1]*uth[i, j-1])/dth*0.5
            d_asth = (Aph[i, j+1]*sth[j+1] - Aph[i, j-1]*sth[j-1])/dth*0.5
         else:
            d_sth_aph = 0.0
            d_buth = 0.0
            d_asth = 0.0

         if inner_i:
            d_rr_aph = (rr[i+1]*Aph[i+1, j] - rr[i-1]*Aph[i-1, j])/drr*0.5
            d_bur = (Bph[i+1, j]*urr[i+1, j]*rr[i+1]
                     - Bph[i-1, j]*urr[i-1, j]*rr[i-1])/drr*0.5
            d_ar = (Aph[i+1, j]*rr[i+1] - Aph[i-1, j]*rr[i-1])/drr*0.5
            d_rb = (rr[i+1]*Bph[i+1, j] - rr[i-1]*Bph[i-1, j])/drr*0.5
         else:
            d_rr_aph = 0.0
            d_bur = 0.0
            d_ar = 0.0
            d_rb = 0.0

         Brr = d_sth_aph/rr_c/sth_c
         Bth = -(d_rr_aph/rr_c)
         Bph_adrr = -(d_bur/rr_c)
         Aph_adrr = -(d_ar*urr[i, j]/rr_c)
         Bph_adth = -(d_buth/rr_c)
         Aph_adth = -(d_asth*uth[i, j]/sth_c/rr_c)

         if has_ip1:
            tb_ip = rrm[i+1]**2*((Bph[i+1, j] - bph_c)/drr)
            ta_ip = rrm[i+1]**2*((Aph[i+1, j] - aph_c)/drr)
            if has_im1:
               tb_i = rrm[i]**2*((bph_c - Bph[i-1, j])/drr)
               ta_i = rrm[i]**2*((aph_c - Aph[i-1, j])/drr)
            else:
               tb_i = rrm[i]**2*0.0
               ta_i = rrm[i]**2*0.0
            div_b_rr = (tb_ip - tb_i)/drr
            div_a_rr = (ta_ip - ta_i)/drr
         else:
            div_b_rr = 0.0
            div_a_rr = 0.0

         if has_jp1:
            tb_jp = sthm[j+1]*((Bph[i, j+1] - bph_c)/dth)
            ta_jp = sthm[j+1]*((Aph[i, j+1] - aph_c)/dth)
            if has_jm1:
               tb_j = sthm[j]*((bph_c - Bph[i, j-1])/dth)
               ta_j = sthm[j]*((aph_c - Aph[i, j-1])/dth)
            else:
               tb_j = sthm[j]*0.0
               ta_j = sthm[j]*0.0
            div_b_th = (tb_jp - tb_j)/dth
            div_a_th = (ta_jp - ta_j)/dth
         else:
            div_b_th = 0.0
            div_a_th = 0.0

         et_c = et[i, j]
         Bph_dfrr = et_c*div_b_rr/rr2
         Bph_dfth = et_c*div_b_th/rr2/sth_c
         Bph_dfex = -(et_c*bph_c/rr2/sth_c**2)
         Aph_dfrr = et_c*div_a_rr/rr2
         Aph_dfth = et_c*div_a_th/rr2/sth_c
         Aph_dfex = -(et_c*aph_c/rr2/sth_c**2)
         Bph_dfrrg = etrr[i, j]*d_rb/rr_c
         Bph_omrr = Brr*omrr[i, j]*rr_c*sth_c
         Bph_omth = Bth*omth[i, j]*rr_c*sth_c

         if nonlocal_alpha:
            b_src = Bph[ibase, j]
         else:
            b_src = bph_c
         Aph_sour = so[i, j]*b_src/(1 + b_src**2)

         dBph = ((Bph_adrr + Bph_adth)
                 + ((Bph_dfrr + Bph_dfth + Bph_dfrrg) + Bph_dfex)
                 + (Bph_omrr + Bph_omth))
         dAph = ((Aph_adrr + Aph_adth)
                 + ((Aph_dfrr + Aph_dfth) + Aph_dfex)
                 + Aph_sour)
         Bphm[i, j] = bph_c + dt*dBph
         Aphm[i, j] = aph_c + dt*dAph

      return Bphm, Aphm

   return kernel


# (fast, nonlocal_alpha) の 4 通りを事前に生成する
_TIME_MARCHING_KERNELS = {}


def get_time_marching_kernel(fast, nonlocal_alpha, separable):
   """(fast, nonlocal_alpha, separable) に対応するカーネルを返す (遅延生成)。"""
   key = (bool(fast), bool(nonlocal_alpha), bool(separable))
   fn = _TIME_MARCHING_KERNELS.get(key)
   if fn is None:
      fn = _make_time_marching_kernel(*key)
      _TIME_MARCHING_KERNELS[key] = fn
   return fn


def _rank1(arr, rtol=1e-13):
   """arr が rank-1 (外積 u(x)v) なら (u, v) を返す。できなければ None。

   背景場 (子午面流・拡散・alpha など) の多くは f(r)*g(theta) の形なので、
   130x130 の 2D 配列ではなく 1D 配列 2 本で表せる。
   """
   i0, j0 = np.unravel_index(np.argmax(np.abs(arr)), arr.shape)
   piv = arr[i0, j0]
   if piv == 0.0:
      if not arr.any():
         return np.zeros(arr.shape[0]), np.zeros(arr.shape[1])
      return None
   u = arr[:, j0]/piv
   v = np.array(arr[i0, :])
   if np.allclose(np.outer(u, v), arr, rtol=rtol, atol=abs(piv)*rtol):
      return np.ascontiguousarray(u), np.ascontiguousarray(v)
   return None


_SEPARABLE_CACHE = {}
_SEPARABLE_FIELDS = ('urr', 'uth', 'et', 'etrr', 'omth', 'so')


def separable_profiles(setup):
   """背景場の rank-1 分解を返す。

   Returns
   -------
   tuple
      (separable, factors)。separable が True なら factors は
      urr_u, urr_v, uth_u, uth_v, et_u, et_v, etrr_u, etrr_v,
      omth_u, omth_v, so_u, so_v の 12 本の 1D 配列。
      False ならダミーのゼロ配列が入る。
   """
   # 背景場は uu0/so0 の更新で変わるので、代表要素で変化を検出する
   fp = tuple(float(getattr(setup, name).flat[k])
              for k, name in enumerate(_SEPARABLE_FIELDS, start=3))
   cached = _SEPARABLE_CACHE.get(id(setup))
   if cached is not None and cached[0] == fp:
      return cached[1]

   factors = []
   separable = True
   for name in _SEPARABLE_FIELDS:
      dec = _rank1(getattr(setup, name))
      if dec is None:
         separable = False
         break
      factors.extend(dec)
   if not separable:
      n_i, n_j = setup.urr.shape
      factors = [np.zeros(n_i), np.zeros(n_j)]*len(_SEPARABLE_FIELDS)
   value = (separable, tuple(factors))
   if len(_SEPARABLE_CACHE) > 8:
      _SEPARABLE_CACHE.clear()
   _SEPARABLE_CACHE[id(setup)] = (fp, value)
   return value


_GRID_1D_CACHE = {}


def _grid_1d(grid):
   """Grid の構造 (RR[i,j]=rr[i], sinTH[i,j]=sin(th[j])) から 1D 配列を作る。

   毎ステップ 2D 配列 4 本を読む代わりに 1D 配列を渡すことで
   メモリトラフィックを減らす。逆数も併せて返す。
   格子は実行中変わらないので、格子形状をキーにキャッシュする。
   """
   key = (grid.ixg, grid.jxg, float(grid.drr), float(grid.dth),
          float(grid.rr[0]), float(grid.th[0]))
   cached = _GRID_1D_CACHE.get(key)
   if cached is not None:
      return cached
   rr = np.ascontiguousarray(grid.RR[:, 0])
   sth = np.ascontiguousarray(grid.sinTH[0, :])
   rrm = np.ascontiguousarray(grid.RRm[:, 0])
   sthm = np.ascontiguousarray(grid.sinTHm[0, :])
   value = (rr, sth, rrm, sthm, 1.0/rr, 1.0/rr**2, 1.0/sth, 1.0/sth**2)
   if len(_GRID_1D_CACHE) > 8:      # 暴走防止 (実運用では格子は1-2種類)
      _GRID_1D_CACHE.clear()
   _GRID_1D_CACHE[key] = value
   return value


@njit(cache=True, boundscheck=False)
def rk_combine(a, b):
   """TVD Runge-Kutta の合成 0.5*a + 0.5*b を一時配列なしで行う。"""
   out = np.empty_like(a)
   for i in range(a.shape[0]):
      for j in range(a.shape[1]):
         out[i, j] = 0.5*a[i, j] + 0.5*b[i, j]
   return out


def time_marching(Bph, Aph, dt, cfg, grid, setup):
   """
   Perform time marching for the magnetic field.

   Parameters
   ----------
   Bph : numpy.ndarray
      Longitudinal magnetic field
   Aph : numpy.ndarray
      Longitudinal vector potential
   dt : float
      Time step
   cfg : S2MFD.Cfg
      Configuration object. ``cfg.exact_arithmetic = True`` を設定すると
      参照実装とビット一致する低速版を使う (既定は高速版)。
   grid : object
      Object containing grid information
   setup : object
      Object containing setup information

   Returns
   -------
   tuple of numpy.ndarray
      - Bphm : Updated longitudinal magnetic field
      - Aphm : Updated longitudinal vector potential

   Notes
   -----
   既定の高速版は除算を逆数の乗算に置き換え、numba の fastmath を有効にする。
   参照実装との差は 1 substep あたり相対 ~2e-16 (倍精度 1 ULP 程度) で、
   物理的な意味を持たない丸め誤差の範囲にある。ビット一致が必要な場合は
   ``cfg.exact_arithmetic = True`` を指定するか
   time_marching_reference() を直接呼ぶこと。
   """
   alpha_code = ALPHA_CODE.get(cfg.alpha_type)
   if alpha_code is None:
      raise ValueError(f"unknown alpha_type: {cfg.alpha_type!r}")

   rr, sth, rrm, sthm, inv_rr, inv_rr2, inv_sth, inv_sth2 = _grid_1d(grid)

   # 非局所 alpha 効果の係数 b/(1+b^2) は j のみに依存するので事前計算する
   if alpha_code == 0:
      b_src = Bph[setup.ibase, :]
   else:
      b_src = np.zeros(Bph.shape[1])
   alpha_fac = b_src/(1 + b_src**2)

   fast = not getattr(cfg, 'exact_arithmetic', False)
   # exact_arithmetic では参照実装とのビット一致を保つため、
   # 背景場は 2D 配列のまま使う (rank-1 再構成は丸めが変わる)
   sep, factors = separable_profiles(setup)
   sep = sep and fast
   kernel = get_time_marching_kernel(fast, alpha_code == 0, sep)
   return kernel(Bph, Aph, dt, rr, sth, rrm, sthm, grid.drr, grid.dth,
                 setup.urr, setup.uth, setup.et, setup.etrr,
                 setup.omrr, setup.omth, setup.so, setup.ibase, alpha_code,
                 inv_rr, inv_rr2, inv_sth, inv_sth2, alpha_fac,
                 np.empty_like(Bph), np.empty_like(Aph), *factors)


def time_marching_reference(Bph, Aph, dt, cfg, grid, setup):
   """
   Reference (unfused) implementation of one Euler substep.

   各項を個別の配列として計算する読みやすい実装。低速だが、
   融合実装 time_marching_fused() の正しさを検証する基準として残してある。

   Parameters / Returns は time_marching() と同じ。
   """
   # calculate poloidal magnetic field
   Brr, Bth = poloidal_mag(Aph, grid.RR, grid.sinTH, grid.drr, grid.dth)
   
   # advection term
   Bph_adrr, Bph_adth, Aph_adrr, Aph_adth \
        = advection(Bph, Aph, grid.RR, grid.sinTH, setup.urr, setup.uth, grid.drr, grid.dth)
        
   # diffusion term
   Bph_dfrr, Bph_dfth, Bph_dfex, Bph_dfrrg, Aph_dfrr, Aph_dfth, Aph_dfex \
          = diffusion(Bph, Aph, grid.RR, grid.sinTH, grid.RRm, grid.sinTHm, grid.drr, grid.dth, setup.et, setup.etrr)   
   # Omega effect
   Bph_omrr, Bph_omth = omega_effect(Brr, Bth, grid.RR, grid.sinTH, setup.omrr, setup.omth)
   
   # source term
   Aph_sour = alpha_effect(Bph, Aph, grid.rr, setup.ibase, setup.so, cfg.alpha_type)
      
   dBph = + (Bph_adrr + Bph_adth) \
          + (Bph_dfrr + Bph_dfth + Bph_dfrrg + Bph_dfex) \
          + (Bph_omrr + Bph_omth) 
          
   dAph = + (Aph_adrr + Aph_adth) \
          + (Aph_dfrr + Aph_dfth + Aph_dfex) \
          + Aph_sour
      
   Bphm = Bph + dt*dBph
   Aphm = Aph + dt*dAph
   
   return Bphm, Aphm
