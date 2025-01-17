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
   if legendre is None:
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
      
         
   else:
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
      # TODO 実際の計算には1~63のみ使用する
      # 必要な配列を初期化
      ant = np.zeros(legendre.termnum-1) # (63,)
      sinth = np.sin(grid.th[grid.margin:grid.jxg - grid.margin])  # θのサイン値(128,)
      sinth_ex = np.repeat(sinth[:, np.newaxis], legendre.termnum-1, axis=1)  # (128, 63)
      P1Sum = np.zeros_like(sinth)  # 合計用配列(128,)

      # Aph, legendre.P1n の一部を事前に切り出し
      Aph_reduced = Aph[grid.ixg - 2, 0:grid.jx]  # Aph の固定行部分(128,)
      # n(0~63まで入っている)
      P1n_reduced = legendre.P1n[1:, :]    # Legendre多項式部分(63,128)
      Aph_ex = np.repeat(Aph_reduced[:, np.newaxis], legendre.termnum-1, axis=1)  # (128, 63)
      # a_n(t) の計算をベクトル化
      # n = 1 から始まるため、スライスで範囲を調整
      n_values = np.arange(1, legendre.termnum)  # n のインデックスを作成(1~63)
      coefficients = (2 * n_values + 1) / (2 * n_values * (n_values + 1))  # 係数部分(63この係数、1~63に対応する)

      # Equation 34: ベクトル化された積分計算(nこ出てきてほしい)
      itg = np.sum(Aph_ex * P1n_reduced[n_values-1, :].T * sinth_ex * grid.dth, axis=0)
      # ant の計算 (ベクトル化済み)
      ant[n_values-1] = coefficients * itg
      ant_ex = np.repeat(ant[:, np.newaxis], grid.jx, axis=1) 
      n_values_ex = np.repeat(ant[:, np.newaxis], grid.jx, axis=1)
      # print(ant.shape)
      # Equation 35: ベクトル化された P1Sum の更新(n=1~63までの足し算をしたい)
      P1Sum = np.sum(
         ((n_values_ex[n_values-1,:] + 1) * ant_ex[n_values-1,:] / cfg.RSUN * P1n_reduced[n_values-1, :]),
         axis=0
      )
      """
      # top boundary condition
      # no electrical current
      # Bph = 0, smoothly match Aph with an exterior potential field solution
      for i in range(0, grid.margin):
         Bph[grid.ixg-i-1,grid.margin:grid.jxg-grid.margin] = -Bph[grid.ixg-2*grid.margin+i,grid.margin:grid.jxg-grid.margin]
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

# # associated legendre polynomials (P^1_n(cosθ))
# def calculate_associated_legendre_polynomials(xx):
#    # associated legendre polynomials (P^1_n(cosθ))
#    # Final term number
#    # grid数で変化するので調べる。
#    termnum = num
#    # n = 0, 1
#    lelist      = np.zeros((len(xx), termnum+1)) 
#    lelist[:,1] = -(1-xx**2)**0.5
#    # n = 2, 3, ..., termnum
#    for l in range(2, termnum+1):
#       # recurrence relation
#       lelist[:,l] = ((2*l-1)/(l-1))*xx*lelist[:,l-1]-(l/(l-1))*lelist[:,l-2]
#    return lelist

# Dikpati+1999
# def top_boundary_condition(Bph, Aph, grid, legendre):
#    PnS = np.zeros_like(grid.TH)
#    ale = np.zeros(legendre.termnum)
#    itg = np.zeros_like(grid.th)

#    # a_n(t)
#    # n are odd numbers to maintain antisymmetry(north⇔south)
#    for i in range(0, legendre.termnum):
#       # integrate (range:0~π)
#       for j in range(grid.margin, grid.jxg - grid.margin):
#          itg[i] += Aph[grid.ixg-grid.margin-1,j]*legendre.P1n[i,j]*np.sin(grid.th[j])* grid.dth
#       # a_n(t)
#       ale[i] = (2*i+1)*cfg.RSUN**(i+1)/(2*i*(i+1))*itg[i]
            
#    # a_n(t)を用いて、(35)式右辺をもとめる
#    for i in range(0, termnum):
#       PnS += -(i+1)*ale[i]/cfg.RSUN**(i+2)*legendre.P1n[i,:]
      
#    # top boundary condition
#    # no electrical current
#    # Bph = 0, smoothly match Aph with an exterior potential field solution
#    # TODO margin=1にのみ対応
#    for i in range(0, grid.margin):
#       Bph[grid.ixg-i-1,grid.margin:grid.jxg-grid.margin] = -Bph[grid.ixg-2*grid.margin+i,grid.margin:grid.jxg-grid.margin]
#       Aph[grid.ixg-i-1,grid.margin:grid.jxg-grid.margin] = +Aph[grid.ixg-i-2,grid.margin:grid.jxg-grid.margin] \
#          -grid.drr*PnS
#       # Aph[grid.ixg-i-1,grid.margin:grid.jxg-grid.margin] = +Aph[grid.ixg-2*grid.margin+i,grid.margin:grid.jxg-grid.margin] \
#       #    -i*grid.drr*PnS
#    return Bph, Aph