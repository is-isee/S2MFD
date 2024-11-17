import numpy as np
import S2MFD

def initialize(cfg):
   import os, json
   
   # make data directory
   if not os.path.isdir(cfg.datadir):
      cfg.cont_flag = False
   os.makedirs(cfg.datadir,exist_ok=True)

   if cfg.cont_flag:
      cfg.load()
   else:
      cfg.save()

   # create grid and setup
   if cfg.cont_flag:
      cfg.load()
      grid = S2MFD.grid_c.load(cfg.gridfile)
      setup = S2MFD.setup_c.load(cfg.setupfile)
   else:
      cfg.save()
      grid = S2MFD.grid_c(ix=cfg.ix,jx=cfg.jx,margin=cfg.margin
               ,rrmin=cfg.rrmin,rrmax=cfg.rrmax,thmin=cfg.thmin,thmax=cfg.thmax)
      grid.save(cfg.gridfile)
      setup = S2MFD.setup_c(cfg,grid)
      setup.save(cfg.setupfile)
   
   return grid, setup

def cfl_condition(grid, setup):
   #CFL condition
   c_cfl=0.8
   dtmin = 1.e10
   for i in range(grid.margin,grid.ixg-grid.margin):
      for j in range(grid.margin,grid.jxg-grid.margin):
         dt_adv = c_cfl * np.min([grid.drr, grid.rr[i]*grid.dth])/ np.sqrt(setup.urr[i,j]**2 + setup.uth[i,j]**2)
         dt_dif = c_cfl * np.min([grid.drr, grid.rr[i]*grid.dth])**2/(2 * setup.et[i,j])
         dtmin = np.min([dtmin,dt_adv,dt_dif])
         
   dt  = dtmin
   return dt

def io(grid, Bph, Aph, time, n, nd):
   Brr, Bth = S2MFD.poloidal_mag(Aph, grid.RR, grid.sinTH, grid.drr, grid.dth)
   print(f"{time/86400:7.1f} [day]; n={n:06d}; nd={nd:04d}")   
   np.savez(file='data/data.'+str(nd).zfill(6)+'.npz' \
               ,Bph=Bph,Aph=Aph,Brr=Brr,Bth=Bth,time=time)

def initial_condition(cfg, grid, setup):
   import glob
   print(cfg.cont_flag)
   # 初期条件
   if cfg.cont_flag:
      files = glob.glob('data/data.*.npz')
      nd = max([int(f.split('.')[-2]) for f in files])
      d = np.load(file='data/data.'+str(nd).zfill(6)+'.npz')
      Aph = d['Aph']
      Bph = d['Bph']
      time = d['time']
      print('yes')
   else:
      nd = 0
      time = 0
      Aph = np.zeros((grid.ixg, grid.jxg))
      Bph = np.zeros((grid.ixg, grid.jxg))
      # Aph = grid.sinTH/(grid.RR/cfg.RSUN)**2
      # Aph[0:setup.ibase,:] = 0
      Bph = np.sin(2*grid.TH)*0.1
      Bph[0:setup.ibase,:] = 0
      

   n = 0

   io(grid, Bph, Aph, time, n, nd)
   
   return Bph, Aph, time, n, nd

def tvd_runge_kutta(Bph, Aph, dt, grid, setup):
   #### dynamo equation                  
   Bphm, Aphm = S2MFD.time_marching(Bph , Aph ,dt, grid, setup)
   Bphm, Aphm = S2MFD.boundary_condition(Bphm, Aphm, grid)

   Bphn, Aphn = S2MFD.time_marching(Bphm, Aphm, dt, grid, setup)
   Bphn, Aphn = S2MFD.boundary_condition(Bphn, Aphn, grid)
   
   Bph = 0.5*Bph + 0.5*Bphn
   Aph = 0.5*Aph + 0.5*Aphn
   
   return Bph, Aph

def main_loop(cfg, grid, setup, Bph, Aph, time, dt, n, nd):
   import matplotlib.pyplot as plt
   
   plt.clf()
   plt.close('all')
   fig = plt.figure('dynamo',figsize=(5,10))   
   
   while time < cfg.tend:
      time += dt
      n += 1
      if(time//cfg.dtout != (time-dt)//cfg.dtout):
         nd += 1
         ax = fig.add_subplot(111,aspect='equal')
         ax.pcolormesh(grid.Y/cfg.RSUN,grid.X/cfg.RSUN,Bph,vmax=5.e0,vmin=-5.e0,cmap='bwr',shading='auto')
         ax.contour(grid.Y/cfg.RSUN,grid.X/cfg.RSUN,grid.RR/cfg.RSUN*grid.sinTH*Aph,colors='black',levels=np.linspace(-8.e10,8.e10,10))
         ax.set_xlim( 0,1)
         ax.set_ylim(-1,1)
         plt.pause(0.01)
         
         io(grid, Bph, Aph, time, n, nd)
      Bph, Aph = tvd_runge_kutta(Bph, Aph, dt, grid, setup)
                        
def run_simulation(cfg=None):
   if cfg is None:
      cfg = S2MFD.config_c()
   
   grid, setup = initialize(cfg)
   dt = cfl_condition(grid, setup)
   Bph, Aph, time, n, nd = initial_condition(cfg, grid, setup)
   main_loop(cfg, grid, setup, Bph, Aph, time, dt, n, nd)
   
# class S2MFD_data(datadir):
#    def __init__(self, cfg, grid, setup, Bph=None, Aph=None, time=None, dt=None, n=None, nd=None):
#       self.cfg = cfg
#       self.grid = grid
#       self.setup = setup
      
#       self.Bph = Bph
#       self.Aph = Aph
#       self.time = time
#       self.dt = dt
#       self.n = n
#       self.nd = nd
      
#    @classmethod
#    def initialize(cls, datadir):
      
#       return  cls(cfg, )
   