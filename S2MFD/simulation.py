import numpy as np
import S2MFD

class Simulation(S2MFD.Data):
    """
    Class for running the simulation inheriting from S2MFD.Data
    """
    def __init__(self, cfg, grid=None, setup=None):
        """
        Parameters
        ----------
        cfg : S2MFD.Cfg
            Configuration object
        grid : S2MFD.Grid, optional
            Grid object
        setup : S2MFD.Setup, optional
            Setup object
        """
        if grid is None:
            grid = S2MFD.Grid(
                ix=cfg.ix, jx=cfg.jx, margin=cfg.margin
               ,rrmin=cfg.rrmin, rrmax=cfg.rrmax
               ,thmin=cfg.thmin, thmax=cfg.thmax
               )
        if setup is None:
            setup = S2MFD.Setup(cfg, grid)
        # TODO Legendreも同様、if文はつけない
        super().__init__(cfg, grid, setup)

    def initialize_simulation(self):
        """
        Initialize the simulation by setting up the grid and setup objects.
        """   
        import os
        
        # make data directory
        if not os.path.isdir(self.cfg.datadir):
            self.cfg.cont_flag = False
            os.makedirs(self.cfg.datadir,exist_ok=True)

        self.cfg.save()

        # create grid and setup
        if self.cfg.cont_flag:
            self.grid = S2MFD.Grid.load(self.cfg.datadir+self.cfg.gridfile)
            self.setup = S2MFD.Setup.load(self.cfg.datadir+self.cfg.setupfile)
        else:
            self.grid.save(self.cfg.datadir+self.cfg.gridfile)
            self.setup.save(self.cfg.datadir+self.cfg.setupfile)
        
    def cfl_condition(self):
        """
        Applies CFL condition
        """
        grid = self.grid
        setup = self.setup
        #CFL condition
        c_cfl=0.8
        dtmin = 1.e10
        for i in range(grid.margin,grid.ixg-grid.margin):
            for j in range(grid.margin,grid.jxg-grid.margin):
                dt_adv = c_cfl * np.min([grid.drr, grid.rr[i]*grid.dth])/ np.sqrt(setup.urr[i,j]**2 + setup.uth[i,j]**2 + 1.e-20)
                dt_dif = c_cfl * np.min([grid.drr, grid.rr[i]*grid.dth])**2/(2 * setup.et[i,j] + 1.e-20)
                dtmin = np.min([dtmin,dt_adv,dt_dif])
                
        self.dt  = dtmin
    
    def save(self):
        """
        Saves data to file
        """
        grid = self.grid
        Brr, Bth = S2MFD.physics.poloidal_mag(self.Aph, grid.RR, grid.sinTH, grid.drr, grid.dth)
        print(f"{self.time/86400:7.1f} [day]; n={self.n:06d}; nd={self.nd:04d}")
        filename = self.get_data_file_path(self.nd)
        np.savez(file=filename \
                    ,Bph=self.Bph,Aph=self.Aph,time=self.time,n=self.n)

    def initial_condition(self):
        """
        Applies initial condition
        """
        cfg = self.cfg
        grid = self.grid
        setup = self.setup
        import glob
        # 初期条件
        if cfg.cont_flag:
            files = glob.glob(cfg.datadir+'data.*.npz')
            self.nd = max([int(f.split('.')[-2]) for f in files])
            self.data_load(self.nd)
        else:
            self.nd = 0
            self.n = 0
            self.time = 0.0
            self.Aph = np.zeros((grid.ixg, grid.jxg))
            self.Bph = np.zeros((grid.ixg, grid.jxg))
            self.Aph = grid.sinTH/(grid.RR/cfg.RSUN)**2*cfg.RSUN/100
            self.Aph[0:setup.ibase,:] = 0
            # data.Bph = np.sin(2*grid.TH)*0.4
            # data.Bph[0:setup.ibase,:] = 0
            
        self.save()
        
    def tvd_runge_kutta(self):
        """
        Applies TVD Runge-Kutta method
        """
        cfg = self.cfg
        grid = self.grid
        setup = self.setup
        #### dynamo equation               
        Bphm, Aphm = S2MFD.physics.time_marching(self.Bph , self.Aph ,self.dt, cfg, grid, setup)
        Bphm, Aphm = S2MFD.physics.boundary_condition(Bphm, Aphm, grid)

        Bphn, Aphn = S2MFD.physics.time_marching(Bphm, Aphm, self.dt, cfg, grid, setup)
        Bphn, Aphn = S2MFD.physics.boundary_condition(Bphn, Aphn, grid)
        
        self.Bph = 0.5*self.Bph + 0.5*Bphn
        self.Aph = 0.5*self.Aph + 0.5*Aphn
    
    def main_loop(self):
        """
        Runs the main loop of the simulation
        """
        import matplotlib.pyplot as plt
        
        cfg = self.cfg
        grid = self.grid
            
        # plt.clf()
        # plt.close('all')
        # fig = plt.figure('dynamo',figsize=(5,10))   
        # ax = fig.add_subplot(111,aspect='equal')
        
        while self.time < cfg.tend:
            self.time += self.dt
            self.n += 1
            if(self.time//cfg.dtout != (self.time - self.dt)//cfg.dtout):
                self.nd += 1
                # ax.clear()
                # ax.pcolormesh(grid.Y/cfg.RSUN,grid.X/cfg.RSUN,data.Bph,vmax=5.e0,vmin=-5.e0,cmap='bwr',shading='auto')
                # ax.contour(grid.Y/cfg.RSUN,grid.X/cfg.RSUN,grid.RR/cfg.RSUN*grid.sinTH*data.Aph/cfg.RSUN,colors='black',levels=np.linspace(-0.02,0.02,16))
                # radius = grid.rrmax/cfg.RSUN
                # ax.plot(radius*np.sin(grid.th),radius*np.cos(grid.th),color='black',alpha=0.4)
                # radius = grid.rrmin/cfg.RSUN         
                # ax.plot(radius*np.sin(grid.th),radius*np.cos(grid.th),color='black',alpha=0.4)
                # ax.set_xlim( 0,1)
                # ax.set_ylim(-1,1)
                # plt.pause(0.01)
                        
                self.save()

            self.tvd_runge_kutta()
            
__all__ = [
         'initialize',
         'cfl_condition',
         'save',
         'initial_condition',
         'tvd_runge_kutta',
         'main_loop',
         ]