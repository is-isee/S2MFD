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
      
         
   elif cfg.boundary_condition_type == 'potential':
      # 時間不変量 (sinth, n_values, coefficients, P1n_reduced) は
      # Legendre.__init__ で事前計算済み
      # 太陽表面 (最外物理セル) の Aφ を Legendre 陪多項式に射影する
      Aph_reduced = Aph[grid.ixg - grid.margin - 1,
                        grid.margin:grid.jxg - grid.margin]  # (jx,)
      # 0~πの積分 (中点則): itg_n = ∫ Aφ P^1_n sinθ dθ
      itg = np.sum(Aph_reduced[None, :] * legendre.P1n_reduced
                   * legendre.sinth[None, :] * grid.dth, axis=1)  # (lmax-1,)
      ant = legendre.coefficients * itg  # (lmax-1,)
      n_col = legendre.n_values[:, None]  # (lmax-1, 1)

      for i in range(0, grid.margin):
         # top boundary condition
         # no electrical current
         # Bph = 0, smoothly match Aph with an exterior potential field solution
         Bph[grid.ixg-i-1,grid.margin:grid.jxg-grid.margin] = -Bph[grid.ixg-2*grid.margin+i,grid.margin:grid.jxg-grid.margin]
         Aph[grid.ixg-i-1,grid.margin:grid.jxg-grid.margin] = np.sum(
            ant[:, None]
            * (grid.rr[grid.ixg - grid.margin - 1]/grid.rr[grid.ixg-i-1])**(n_col + 1)
            * legendre.P1n_reduced, axis=0)

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