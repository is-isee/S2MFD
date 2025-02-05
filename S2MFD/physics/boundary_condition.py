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
      # for文（非常に時間がかかる）
      """
      ant = np.zeros(legendre.termnum)
      sinth = np.sin(grid.th[grid.margin:grid.jxg-grid.margin])
      itg = np.zeros(legendre.termnum)
      P1Sum = np.zeros_like(sinth)

      # a_n(t)
      # n are odd numbers to maintain antisymmetry(north⇔south)
      for n in range(1, legendre.termnum):
         # 34式
         # integrate (range:0~π)
         for j in range(0, grid.jx):
            itg[n] += Aph[grid.ixg-2,j]*legendre.P1n[n,j]*sinth[j]* grid.dth
         ant[n] = (2*n+1)/(2*n*(n+1))*itg[n]
         
         # 35式
         P1Sum += (n+1)*ant[n]/cfg.RSUN*legendre.P1n[n]
      """
      # for文を用いない。meshgrid
      # (n,θ)に統一
      """
      # 必要な配列を初期化
      ant = np.zeros(legendre.lmax-1) # (127(n),)
      RSUN_dim = cfg.RSUN
      sinth = np.sin(grid.th[grid.margin:grid.jxg - grid.margin])  # θのサイン値(128(θ),)
      n_values = np.arange(1, legendre.lmax)  # n のインデックスを作成(1~127)
      Aph_reduced = Aph[grid.ixg - 2, 1:grid.jx+1]  # 太陽表面のAφ(128(θ),)、1~128
      P1Sum = np.zeros_like(sinth)  # 合計用配列(128(θ),)
      P1n_reduced = legendre.P1n[1:, :]   # (127(n), 128(θ))
      
      sinth_ex, n_values_ex  = np.meshgrid(sinth, n_values) # (127(n), 128(θ))
      Aph_ex, n_values_ex  = np.meshgrid(Aph_reduced, n_values) # (127(n), 128(θ))
      
      # coefficients = (2 * n_values + 1) / (2 * n_values * (n_values + 1))*RSUN_dim**(n_values+1)  # (127(n),)
      coefficients = (2 * n_values + 1) / (2 * n_values * (n_values + 1)) # (127(n),)
      # 0~πの積分
      itg = np.sum(Aph_ex * P1n_reduced * sinth_ex * grid.dth, axis=1) # (127(n),)
      # ant の計算 (ベクトル化済み)
      ant = coefficients * itg # (127(n),)
      # θを追加
      ant_ex, dammy  = np.meshgrid(ant, sinth, indexing = 'ij') # (127(n), 128(θ))
      P1Sum = np.sum((n_values_ex + 1) * ant_ex / RSUN_dim * P1n_reduced,axis=0)
      """
      # for文を用いない。not_meshgrid
      # 必要な配列を初期化
      ant = np.zeros(legendre.lmax-1) # (127(n),)
      sinth = np.sin(grid.th[grid.margin:grid.jxg - grid.margin])  # θのサイン値(128(θ),)
      sinth_ex = np.repeat(sinth[:, np.newaxis], legendre.lmax-1, axis=1)  # (128(θ), 127(n))
      P1Sum = np.zeros_like(sinth)  # 合計用配列(128(θ),)
      RSUN_dim = cfg.RSUN # 1
      # Aph, legendre.P1n の一部を事前に切り出し
      Aph_reduced = Aph[grid.ixg - 2, 1:grid.jx+1]  # 太陽表面のAφ(128(θ),)、1~128
      # n(0~63まで入っている)
      P1n_reduced = legendre.P1n[1:, :].T   # (128(θ),127(n))
      Aph_ex = np.repeat(Aph_reduced[:, np.newaxis], legendre.lmax-1, axis=1)  # (128(θ),127(n))
      # a_n(t) の計算をベクトル化
      # n = 1 から始まるため、スライスで範囲を調整
      n_values = np.arange(1, legendre.lmax)  # n のインデックスを作成(1~127)
      # coefficients = (2 * n_values + 1) / (2 * n_values * (n_values + 1))*RSUN_dim**(n_values+1)  # (127(n),)
      coefficients = (2 * n_values + 1) / (2 * n_values * (n_values + 1)) # (127(n),)
      # Equation 34: ベクトル化された積分計算(nこ出てきてほしい)
      itg = np.sum(Aph_ex * P1n_reduced * sinth_ex * grid.dth, axis=0) # (127(n),)
     
      # ant の計算 (ベクトル化済み)
      ant = coefficients * itg # (127(n),)
      # θを追加
      ant_ex = np.repeat(ant[:, np.newaxis], grid.jx, axis=1) # (127(n),128(θ))
      # θを追加
      n_values_ex = np.repeat(n_values[:, np.newaxis], grid.jx, axis=1) # (127(n),128(θ))
      # Equation 35: ベクトル化された P1Sum の更新(n=1~63までの足し算をしたい)
      # P1Sum = np.sum(
      #    ((n_values_ex + 1) * ant_ex / RSUN_dim**(n_values_ex + 2) * P1n_reduced.T),
      #    axis=0
      # )
      P1Sum = np.sum((n_values_ex + 1) * ant_ex / RSUN_dim * P1n_reduced.T,axis=0)
      for i in range(0, grid.margin):
         # top boundary condition
         # no electrical current
         # Bph = 0, smoothly match Aph with an exterior potential field solution
         Bph[grid.ixg-i-1,grid.margin:grid.jxg-grid.margin] = -Bph[grid.ixg-2*grid.margin+i,grid.margin:grid.jxg-grid.margin]
         # print(P1Sum) クソデカ
         Aph[grid.ixg-i-1,grid.margin:grid.jxg-grid.margin] = +Aph[grid.ixg-i-2,grid.margin:grid.jxg-grid.margin] \
            -grid.drr*P1Sum
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