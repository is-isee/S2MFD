import numpy as np
import S2MFD

class Simulation(S2MFD.Data):
    """
    Class for running the simulation inheriting from S2MFD.Data
    """
    def __init__(self, cfg, grid=None, setup=None, legendre=None):
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
            
        legendre = S2MFD.Legendre(grid)
        super().__init__(cfg, grid, setup ,legendre)

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
            self.legendre = S2MFD.Legendre.load(self.cfg.datadir+self.cfg.legendrefile)
        else:
            self.grid.save(self.cfg.datadir+self.cfg.gridfile)
            self.setup.save(self.cfg.datadir+self.cfg.setupfile)
            self.legendre.save(self.cfg.datadir+self.cfg.legendrefile)
        
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
                    ,Bph=self.Bph,Aph=self.Aph,time=self.time,n=self.n,uu0=self.cfg.uu0,so0=self.cfg.so0)

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
            
        self.cfg.so0 = cfg.so0_time_dependent(self.time, cfg.ett, cfg.RSUN)
        self.cfg.uu0 = cfg.uu0_time_dependent(self.time, cfg.ett, cfg.RSUN)
        self.setup = S2MFD.Setup(self.cfg, grid)
        self.save()
        
    def initial_for_bisection(self):
        """
        Applies initial condition
        """
        cfg = self.cfg
        grid = self.grid
        setup = self.setup

        self.nd = 0
        self.n = 0
        self.time = 0.0
        self.Aph = np.zeros((grid.ixg, grid.jxg))
        self.Bph = np.zeros((grid.ixg, grid.jxg))
        self.Aph = grid.sinTH/(grid.RR/cfg.RSUN)**2*cfg.RSUN/100
        self.Aph[0:setup.ibase,:] = 0

        self.save()
        
    def tvd_runge_kutta(self):
        """
        Applies TVD Runge-Kutta method
        """
        cfg = self.cfg
        grid = self.grid
        setup = self.setup
        legendre = self.legendre
        #### dynamo equation               
        Bphm, Aphm = S2MFD.physics.time_marching(self.Bph , self.Aph ,self.dt, cfg, grid, setup)
        Bphm, Aphm = S2MFD.physics.boundary_condition(Bphm, Aphm, cfg, grid, legendre)

        Bphn, Aphn = S2MFD.physics.time_marching(Bphm, Aphm, self.dt, cfg, grid, setup)
        Bphn, Aphn = S2MFD.physics.boundary_condition(Bphn, Aphn, cfg, grid, legendre)
        
        self.Bph = 0.5*self.Bph + 0.5*Bphn
        self.Aph = 0.5*self.Aph + 0.5*Aphn

    def tvd_runge_kutta_bisection(self,Bph_df, Aph_df):
        """
        Applies TVD Runge-Kutta method
        """
        cfg = self.cfg
        grid = self.grid
        setup = self.setup
        legendre = self.legendre
        #### dynamo equation               
        for _ in range(cfg.dtout//cfg.d2s):
            Bphm, Aphm = S2MFD.physics.time_marching(Bph_df , Aph_df ,self.dt, cfg, grid, setup)
            Bphm, Aphm = S2MFD.physics.boundary_condition(Bphm, Aphm, cfg, grid, legendre)

            Bphn, Aphn = S2MFD.physics.time_marching(Bphm, Aphm, self.dt, cfg, grid, setup)
            Bphn, Aphn = S2MFD.physics.boundary_condition(Bphn, Aphn, cfg, grid, legendre)
            
            Bph_df = 0.5*Bph_df + 0.5*Bphn
            Aph_df = 0.5*Aph_df + 0.5*Aphn
            
        return Bph_df, Aph_df
    
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
        
        # Real Time Butterfly Diagram
        # plt.clf()
        # plt.close('all')
        # fig = plt.figure('Butterfly Diagram',figsize=(10,5))
        # ax = fig.add_subplot(1,1,1)
        # Bpht_b = np.zeros((grid.jxg,cfg.tend//cfg.dtout))
        # time = np.linspace(0,cfg.tend,cfg.tend//cfg.dtout)
        
        while self.time < cfg.tend:
            self.time += self.dt
            self.n += 1
            if(self.time//cfg.dtout != (self.time - self.dt)//cfg.dtout):
                self.nd += 1
                
                # magnetic field
                # ax.clear()
                # ax.pcolormesh(grid.Y/cfg.RSUN,grid.X/cfg.RSUN,self.Bph,vmax=5.e0,vmin=-5.e0,cmap='bwr',shading='auto')
                # ax.contour(grid.Y/cfg.RSUN,grid.X/cfg.RSUN,grid.RR/cfg.RSUN*grid.sinTH*self.Aph/cfg.RSUN,colors='black',levels=np.linspace(-0.02,0.02,16))
                # radius = grid.rrmax/cfg.RSUN
                # ax.plot(radius*np.sin(grid.th),radius*np.cos(grid.th),color='black',alpha=0.4)
                # radius = grid.rrmin/cfg.RSUN         
                # ax.plot(radius*np.sin(grid.th),radius*np.cos(grid.th),color='black',alpha=0.4)
                # ax.set_xlim( 0,1)
                # ax.set_ylim(-1,1)
                # plt.pause(0.01)
                
                # Real Time Butterfly Diagram
                # ax.clear()
                # Bpht_b[:,self.nd-1] = self.Bph[1+np.argmin(abs(grid.rr-0.7*cfg.RSUN)),:]
                # ax.pcolormesh(time,grid.th/np.pi*180,Bpht_b,cmap='bwr',shading='auto')
                # mask_p = Bpht_b > 3.5
                # mask_m = Bpht_b < -3.5
                # y_vals, x_vals = np.where(mask_p)
                # ax.scatter(time[x_vals], (grid.th / np.pi * 180)[y_vals], color='black', s=1, label='>3.5')
                # y_vals, x_vals = np.where(mask_m)
                # ax.scatter(time[x_vals], (grid.th / np.pi * 180)[y_vals], color='black', s=1, label='<-3.5')
                # ax.contourf(time, grid.th/np.pi*180, mask_p, levels=[0.5, 1.5], colors=['black'])
                # ax.contourf(time, grid.th/np.pi*180, mask_m, levels=[0.5, 1.5], colors=['black'])
                # plt.xlim(self.time-3600*self.dt,self.time)
                # plt.pause(0.01)
                # print(cfg.boundary_condition_type)
                
                # 時間依存の so0 と uu0 を計算
                self.cfg.so0 = cfg.so0_time_dependent(self.time, cfg.ett, cfg.RSUN)
                self.cfg.uu0 = cfg.uu0_time_dependent(self.time, cfg.ett, cfg.RSUN)
                self.setup = S2MFD.Setup(self.cfg, grid)
                self.save()

            self.tvd_runge_kutta()

    def main_loop_for_bisection(self, Sunspot_N, uu0t):
        """
        Runs the main loop of the simulation
        """
        import matplotlib.pyplot as plt
        
        cfg = self.cfg
        grid = self.grid
        
        #　許容残差
        eps = 0.0001
        
        # 初期値
        umin = 0
        umax = 3000
        obsn = 0
        
        # 表の作成
        import pandas as pd
        col_names = ['u0_ans', 'u0', '誤差']
        df = pd.DataFrame(columns=col_names)

        while self.time < cfg.tend:
            obsn += 1
            obs = Sunspot_N[self.n]
            def delta(now):
                return obs - now
            while True:
                umid = (umin + umax)/2
                
                # aの計算
                Bph_dfa = self.Bph
                Aph_dfa = self.Aph
                self.cfg.uu0 = umin
                self.setup = S2MFD.Setup(self.cfg, grid)
                Bph_dfa, Aph_dfa = self.tvd_runge_kutta_bisection(Bph_dfa, Aph_dfa)
                a_sim = self.snumbers_for_bisection(Bph_dfa)
                
                # bの計算
                Bph_dfb = self.Bph
                Aph_dfb = self.Aph
                self.cfg.uu0 = umax
                self.setup = S2MFD.Setup(self.cfg, grid)
                Bph_dfb, Aph_dfb = self.tvd_runge_kutta_bisection(Bph_dfb, Aph_dfb)
                b_sim = self.snumbers_for_bisection(Bph_dfb)
                
                # cの計算
                Bph_dfc = self.Bph
                Aph_dfc = self.Aph
                self.cfg.uu0 = umid
                self.setup = S2MFD.Setup(self.cfg, grid)
                Bph_dfc, Aph_dfc = self.tvd_runge_kutta_bisection(Bph_dfc, Aph_dfc)
                c_sim = self.snumbers_for_bisection(Bph_dfc)

                if delta(a_sim) * delta(c_sim) < 0:
                    umax = umid
                else:
                    umin = umid
                if abs(delta(c_sim)) < eps:
                    self.Bph = Bph_dfc
                    self.Aph = Aph_dfc
                    self.cfg.uu0 = umid
                    self.setup = S2MFD.Setup(self.cfg, grid)
                    self.time += (cfg.dtout//cfg.d2s) * self.dt
                    self.n += cfg.dtout//cfg.d2s
                    self.save()
                    break
    
    # ========================================================================================== #
    # 二分法シミュレーションのための黒点数を数えるコード
    def snumbers_for_bisection(self,Bpht):
        # 黒点数の計上
        thrsh = 3.0
        base  = 1+np.argmin(abs(self.grid.rr-0.7*self.cfg.RSUN))
        stat  = 1+np.argmin(abs(self.grid.th- 40/180*np.pi))
        endd  = 1+np.argmin(abs(self.grid.th-140/180*np.pi))
        # グリッドに応じた北半球と南半球の分割
        if self.grid.jxg % 2==0: 
            S_equa = self.grid.jxg//2 - 1
            N_equa = S_equa + 1
        else:
            S_equa = (self.grid.jxg-1)//2 - 1
            N_equa = S_equa + 2

        # S_numとN_numは全く同じになる（南北対称だから当たり前）
        # n1は時間要素の最後の番号
        S_num = np.sum(Bpht[base,stat:S_equa+1]<-thrsh) + np.sum(Bpht[base,stat:S_equa+1]>thrsh)
        N_num = np.sum(Bpht[base,N_equa:endd]<-thrsh)   + np.sum(Bpht[base,N_equa:endd]>thrsh)
        # 観測に基づいた調整パラメタ
        kappa = 0.3
        S_num = kappa * S_num
        N_num = kappa * N_num
        
        return S_num
    # ========================================================================================== #

# TODO __all__の中身に追加した関数を加える 
__all__ = [
         'initialize',
         'cfl_condition',
         'save',
         'initial_condition',
         'tvd_runge_kutta',
         'main_loop',
         ]