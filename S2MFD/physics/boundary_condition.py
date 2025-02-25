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
      - Bph : Radial  magnetic field
      - Aph : Latitudinal magnetic
      
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
      ant = np.zeros(legendre.lmax-1) # (127(n),)
      sinth = np.sin(grid.th[grid.margin:grid.jxg - grid.margin])  # θのサイン値(128(θ),)
      n_values = np.arange(1, legendre.lmax)  # n のインデックスを作成(1~127)
      
      Aph_reduced = Aph[grid.ixg - 2, 1:grid.jx+1]  # 太陽表面のAφ(128(θ),)、1~128
      P1n_reduced = legendre.P1n[1:, :]   # (127(n), 128(θ))
      
      n_values_ex, sinth_ex  = np.meshgrid(n_values, sinth, indexing = 'ij') # (127(n), 128(θ))
      n_values_ex, Aph_ex    = np.meshgrid(n_values, Aph_reduced,indexing = 'ij') # (127(n), 128(θ))
      
      coefficients = (2 * n_values + 1) / (2 * n_values * (n_values + 1)) # (127(n),)
      # 0~πの積分
      itg = np.sum(Aph_ex * P1n_reduced * sinth_ex * grid.dth, axis=1) # (127(n),)
      # ant の計算 (ベクトル化済み)
      ant = coefficients * itg # (127(n),)
      # θを追加
      ant_ex, dammy  = np.meshgrid(ant, sinth, indexing = 'ij') # (127(n), 128(θ))

      for i in range(0, grid.margin):
         # top boundary condition
         # no electrical current
         # Bph = 0, smoothly match Aph with an exterior potential field solution
         Bph[grid.ixg-i-1,grid.margin:grid.jxg-grid.margin] = -Bph[grid.ixg-2*grid.margin+i,grid.margin:grid.jxg-grid.margin]
         # Aph[grid.ixg-i-1,grid.margin:grid.jxg-grid.margin] = +Aph[grid.ixg-i-2,grid.margin:grid.jxg-grid.margin] \
         #    -grid.drr*np.sum((n_values_ex + 1) * ant_ex * (grid.rr[grid.ixg-grid.margin-1]/grid.rr[grid.ixg-i-1])**(n_values_ex + 1) * (1/grid.rr[grid.ixg-i-1]) * P1n_reduced,axis=0)
         Aph[grid.ixg-i-1,grid.margin:grid.jxg-grid.margin] = np.sum(ant_ex * (grid.rr[grid.ixg - grid.margin - 1]/grid.rr[grid.ixg-i-1])**(n_values_ex + 1) * P1n_reduced,axis=0)
      
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