#
def boundary_condition(Bph, Aph, grid):
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
   # 動径方向境界条件
   for i in range(0, grid.margin):
      # bottom boundary condition
      # perfect conductor
      # Aph = 0, d(r*Bph)/dr = 0
      Aph[i,grid.margin:grid.jxg-grid.margin] = - Aph[2*grid.margin-i-1,grid.margin:grid.jxg-grid.margin]
      Bph[i,grid.margin:grid.jxg-grid.margin] = + Bph[2*grid.margin-i-1,grid.margin:grid.jxg-grid.margin] \
         /grid.rr[i]*grid.rr[2*grid.margin-i-1]

      # top boundary condition 
      # radial condition 
      # Bph = 0, d(r*Aph)/dr = 0
      
      Bph[grid.ixg-i-1,grid.margin:grid.jxg-grid.margin] = -Bph[grid.ixg-2*grid.margin+i,grid.margin:grid.jxg-grid.margin]
      Aph[grid.ixg-i-1,grid.margin:grid.jxg-grid.margin] = +Aph[grid.ixg-2*grid.margin+i,grid.margin:grid.jxg-grid.margin] \
         /grid.rr[grid.ixg-i-1]*grid.rr[grid.ixg-2*grid.margin+i]

   # 緯度方向境界条件      
   for j in range(0, grid.margin):
      #pole A=B=0
      Bph[grid.margin:grid.ixg-grid.margin,j]  = -Bph[grid.margin:grid.ixg-grid.margin,2*grid.margin-j-1]
      Aph[grid.margin:grid.ixg-grid.margin,j]  = -Aph[grid.margin:grid.ixg-grid.margin,2*grid.margin-j-1]

      Bph[grid.margin:grid.ixg-grid.margin,grid.jxg-j-1] = -Bph[grid.margin:grid.ixg-grid.margin,grid.jxg-2*grid.margin+j]
      Aph[grid.margin:grid.ixg-grid.margin,grid.jxg-j-1] = -Aph[grid.margin:grid.ixg-grid.margin,grid.jxg-2*grid.margin+j]
   return Bph, Aph

# associated legendre polynomials (P^1_n(cosθ))
def calculate_associated_legendre_polynomials(xx):
   # associated legendre polynomials (P^1_n(cosθ))
   # n = 0, 1
   lelist = [np.zeros_like(xx), -(1-xx**2)**0.5]
   # Final term number
   termnum = 7
   # n = 2, 3, ..., termnum
   for l in range(2, termnum+1):
      # recurrence relation
      alp = ((2*l-1)/(l-1))*xx*lelist[l-1]-(l/(l-1))*lelist[l-2]
      # associated legendre polynomials list
      lelist.append(alp)
   return lelist

# Dikpati+1999
def top_boundary_condition(Bph, Aph, grid, lelist):
   PnS = np.zeros_like(grid.TH)
   ale = np.zeros(termnum + 1)
   integral = np.zeros_like(grid.th)

   # a_n(t)
   # n are odd numbers to maintain antisymmetry(north⇔south)
   for i in range(0, termnum+1):
      if i % 2 != 0:
         # integrand
         integrand = Aph[grid.ixg-grid.margin-1,:]*lelist[i]*grid.sinTH
         #  TODO integrate (range:0~π/2)
         # a_n(t)
         for j in range(grid.margin, grid.jxg - grid.margin):
            ale[i] += (2*i+1)*cfg.RSUN**(i+1)/(i*(i+1))*integrand[j] * grid.dth
      else:
         ale[i] = 0
            
   # a_n(t)を用いて、(35)式右辺をもとめる
   for i in range(0, termnum+1):
      PnS += -(i+1)*ale[i]/cfg.RSUN**(i+2)*lelist[i]
      
   # top boundary condition
   # no electrical current
   # Bph = 0, smoothly match Aph with an exterior potential field solution
   # TODO margin=1にのみ対応
   for i in range(0, grid.margin):
      Bph[grid.ixg-i-1,grid.margin:grid.jxg-grid.margin] = -Bph[grid.ixg-2*grid.margin+i,grid.margin:grid.jxg-grid.margin]
      Aph[grid.ixg-i-1,grid.margin:grid.jxg-grid.margin] = +Aph[grid.ixg-i-2,grid.margin:grid.jxg-grid.margin] \
         -grid.drr*PnS
      # Aph[grid.ixg-i-1,grid.margin:grid.jxg-grid.margin] = +Aph[grid.ixg-2*grid.margin+i,grid.margin:grid.jxg-grid.margin] \
      #    -i*grid.drr*PnS
   return Bph, Aph