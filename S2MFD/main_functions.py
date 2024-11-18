import numpy as np
import S2MFD

def initialize(cfg):
   """
   Initialize the simulation by setting up the grid and setup objects.

   Parameters:
   cfg (object): S2MFD.config_c object

   Returns:
   object: Initialized S2MFD.data object containing configuration, grid, and setup.
   """   
   import os
   
   # make data directory
   if not os.path.isdir(cfg.datadir):
      cfg.cont_flag = False
      os.makedirs(cfg.datadir,exist_ok=True)

   cfg.save()

   # create grid and setup
   if cfg.cont_flag:
      grid = S2MFD.grid_c.load(cfg.gridfile)
      setup = S2MFD.setup_c.load(cfg.setupfile)
   else:
      grid = S2MFD.grid_c(ix=cfg.ix,jx=cfg.jx,margin=cfg.margin
               ,rrmin=cfg.rrmin,rrmax=cfg.rrmax,thmin=cfg.thmin,thmax=cfg.thmax)
      grid.save(cfg.gridfile)
      setup = S2MFD.setup_c(cfg,grid)
      setup.save(cfg.setupfile)
      
   data = S2MFD.S2MFD_data(cfg,grid,setup)
      
   return data

def cfl_condition(data):
   grid = data.grid
   setup = data.setup
   #CFL condition
   c_cfl=0.8
   dtmin = 1.e10
   for i in range(grid.margin,grid.ixg-grid.margin):
      for j in range(grid.margin,grid.jxg-grid.margin):
         dt_adv = c_cfl * np.min([grid.drr, grid.rr[i]*grid.dth])/ np.sqrt(setup.urr[i,j]**2 + setup.uth[i,j]**2 + 1.e-20)
         dt_dif = c_cfl * np.min([grid.drr, grid.rr[i]*grid.dth])**2/(2 * setup.et[i,j] + 1.e-20)
         dtmin = np.min([dtmin,dt_adv,dt_dif])
         
   data.dt  = dtmin
   
   # cfg = data.cfg
   # data.dt = 5.e-6*cfg.RSUN**2/cfg.ett
   # print(data.dt)
   return data

def io(data):
   grid = data.grid
   Brr, Bth = S2MFD.poloidal_mag(data.Aph, grid.RR, grid.sinTH, grid.drr, grid.dth)
   print(f"{data.time/86400:7.1f} [day]; n={data.n:06d}; nd={data.nd:04d}")
   filename = data.get_data_file_path(data.nd)
   np.savez(file=filename \
               ,Bph=data.Bph,Aph=data.Aph,time=data.time,n=data.n)

def initial_condition(data):
   cfg = data.cfg
   grid = data.grid
   setup = data.setup
   import glob
   # 初期条件
   if cfg.cont_flag:
      files = glob.glob('data/data.*.npz')
      data.nd = max([int(f.split('.')[-2]) for f in files])
      data.data_load(data.nd)
   else:
      data.nd = 0
      data.n = 0
      data.time = 0.0
      data.Aph = np.zeros((grid.ixg, grid.jxg))
      data.Bph = np.zeros((grid.ixg, grid.jxg))
      data.Aph = grid.sinTH/(grid.RR/cfg.RSUN)**2*cfg.RSUN/100
      data.Aph[0:setup.ibase,:] = 0
      # data.Bph = np.sin(2*grid.TH)*0.4
      # data.Bph[0:setup.ibase,:] = 0
      
   io(data)
   
   return data

def tvd_runge_kutta(data):
   cfg = data.cfg
   grid = data.grid
   setup = data.setup
   #### dynamo equation               
   Bphm, Aphm = S2MFD.time_marching(data.Bph , data.Aph ,data.dt, cfg, grid, setup)
   Bphm, Aphm = S2MFD.boundary_condition(Bphm, Aphm, grid)

   Bphn, Aphn = S2MFD.time_marching(Bphm, Aphm, data.dt, cfg, grid, setup)
   Bphn, Aphn = S2MFD.boundary_condition(Bphn, Aphn, grid)
   
   data.Bph = 0.5*data.Bph + 0.5*Bphn
   data.Aph = 0.5*data.Aph + 0.5*Aphn
   
   return data

def main_loop(data):
   import matplotlib.pyplot as plt
   
   cfg = data.cfg
   grid = data.grid
      
   plt.clf()
   plt.close('all')
   fig = plt.figure('dynamo',figsize=(5,10))   
   ax = fig.add_subplot(111,aspect='equal')
   
   while data.time < cfg.tend:
      data.time += data.dt
      data.n += 1
      if(data.time//cfg.dtout != (data.time-data.dt)//cfg.dtout):
         data.nd += 1
         ax.clear()
         ax.pcolormesh(grid.Y/cfg.RSUN,grid.X/cfg.RSUN,data.Bph,vmax=5.e0,vmin=-5.e0,cmap='bwr',shading='auto')
         ax.contour(grid.Y/cfg.RSUN,grid.X/cfg.RSUN,grid.RR/cfg.RSUN*grid.sinTH*data.Aph/cfg.RSUN,colors='black',levels=np.linspace(-0.01,0.01,20))
         ax.set_xlim( 0,1)
         ax.set_ylim(-1,1)
         plt.pause(0.01)
                  
         io(data)

      data = tvd_runge_kutta(data)
                              
def run_simulation(cfg=None, parameter_file=None):
   if cfg is None:
      if parameter_file is None:
         cfg = S2MFD.config_c()
      else:
         cfg = S2MFD.config_c(parameter_file)
   
   data = initialize(cfg)
   cfl_condition(data)
   initial_condition(data)
   main_loop(data)