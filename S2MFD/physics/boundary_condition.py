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