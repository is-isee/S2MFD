import numpy as np

def boundary_condition(Bph, Aph, cfg, grid, legendre):
   """
   Applies boundary condition for the magnetic field.
   
   Parameters
   ----------
   Bph : numpy.ndarray
      Longitudinal magnetic field
   Aph : numpy.ndarray
      Longitudinal vector potential
      
   Returns
   -------
   tuple of numpy.ndarray
      - Bph : Longitudinal magnetic field (ghost cells filled)
      - Aph : Longitudinal vector potential (ghost cells filled)


   Notes
   -----
   We do not have to return Bph and Aph because they are mutable objects, but we do so for clarity.
   """
   if cfg.boundary_condition_type == 'vertical':
      # 動径方向境界条件
      for i in range(0, grid.margin):
         # top boundary condition 
         # radial condition 
         # Bph = 0, d(r*Aph)/dr = 0
         Bph[grid.ixg-i-1,grid.margin:grid.jxg-grid.margin] = -Bph[grid.ixg-2*grid.margin+i,grid.margin:grid.jxg-grid.margin]
         Aph[grid.ixg-i-1,grid.margin:grid.jxg-grid.margin] = +Aph[grid.ixg-2*grid.margin+i,grid.margin:grid.jxg-grid.margin] \
            /grid.rr[grid.ixg-i-1]*grid.rr[grid.ixg-2*grid.margin+i]
         # bottom boundary condition
         # perfect conductor
         # Aph = 0, d(r*Bph)/dr = 0
         Aph[i,grid.margin:grid.jxg-grid.margin] = - Aph[2*grid.margin-i-1,grid.margin:grid.jxg-grid.margin]
         Bph[i,grid.margin:grid.jxg-grid.margin] = + Bph[2*grid.margin-i-1,grid.margin:grid.jxg-grid.margin] \
            /grid.rr[i]*grid.rr[2*grid.margin-i-1]
      
         
   elif cfg.boundary_condition_type == 'R06':
      # Rempel (2006) §2.2:
      #   "B_Phi vanishes at both radial boundaries, while A vanishes at the
      #    inner boundary, and the poloidal field is assumed to be radial at
      #    the top boundary."
      # 'vertical' との違いは下部境界の B_phi だけ (完全導体 d(rB)/dr=0 ではなく
      # B_phi = 0)。深部は磁気拡散が 3 桁小さいので、ここの扱いが効く。
      for i in range(0, grid.margin):
         # 上部: B_phi = 0, ポロイダル場は動径方向 (d(rA)/dr = 0)
         Bph[grid.ixg-i-1,grid.margin:grid.jxg-grid.margin] = -Bph[grid.ixg-2*grid.margin+i,grid.margin:grid.jxg-grid.margin]
         Aph[grid.ixg-i-1,grid.margin:grid.jxg-grid.margin] = +Aph[grid.ixg-2*grid.margin+i,grid.margin:grid.jxg-grid.margin] \
            /grid.rr[grid.ixg-i-1]*grid.rr[grid.ixg-2*grid.margin+i]
         # 下部: A = 0, B_phi = 0
         Aph[i,grid.margin:grid.jxg-grid.margin] = - Aph[2*grid.margin-i-1,grid.margin:grid.jxg-grid.margin]
         Bph[i,grid.margin:grid.jxg-grid.margin] = - Bph[2*grid.margin-i-1,grid.margin:grid.jxg-grid.margin]

   elif cfg.boundary_condition_type == 'potential':
      # ルジャンドル射影と外部ポテンシャル場による再構成は Aφ について線形なので、
      # Legendre.build_potential_operator() が両者をまとめた行列を用意している。
      # ここでは行列ベクトル積 1 回で済む。
      Aph_reduced = Aph[grid.ixg - grid.margin - 1,
                        grid.margin:grid.jxg - grid.margin]  # (jx,)

      for i in range(0, grid.margin):
         # top boundary condition
         # no electrical current
         # Bph = 0, smoothly match Aph with an exterior potential field solution
         Bph[grid.ixg-i-1,grid.margin:grid.jxg-grid.margin] = -Bph[grid.ixg-2*grid.margin+i,grid.margin:grid.jxg-grid.margin]
         Aph[grid.ixg-i-1,grid.margin:grid.jxg-grid.margin] = \
            Aph_reduced @ legendre.potential_operator[i]

         # bottom boundary condition
         # perfect conductor
         # Aph = 0, d(r*Bph)/dr = 0
         Aph[i,grid.margin:grid.jxg-grid.margin] = - Aph[2*grid.margin-i-1,grid.margin:grid.jxg-grid.margin]
         Bph[i,grid.margin:grid.jxg-grid.margin] = + Bph[2*grid.margin-i-1,grid.margin:grid.jxg-grid.margin] \
            /grid.rr[i]*grid.rr[2*grid.margin-i-1]

   # 緯度方向境界条件      
   for j in range(0, grid.margin):
      #pole A=B=0
      Bph[grid.margin:grid.ixg-grid.margin,j]  = -Bph[grid.margin:grid.ixg-grid.margin,2*grid.margin-j-1]
      Aph[grid.margin:grid.ixg-grid.margin,j]  = -Aph[grid.margin:grid.ixg-grid.margin,2*grid.margin-j-1]

      Bph[grid.margin:grid.ixg-grid.margin,grid.jxg-j-1] = -Bph[grid.margin:grid.ixg-grid.margin,grid.jxg-2*grid.margin+j]
      Aph[grid.margin:grid.ixg-grid.margin,grid.jxg-j-1] = -Aph[grid.margin:grid.ixg-grid.margin,grid.jxg-2*grid.margin+j]
   return Bph, Aph