from typing import Dict
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
        # print(f"{self.time/86400:7.1f} [day]; n={self.n:06d}; nd={self.nd:04d}")
        filename = self.get_data_file_path(self.nd)
        np.savez(file=filename \
                    ,Bph=self.Bph,Aph=self.Aph,time=self.time,n=self.n,nd=self.nd,uu0=self.cfg.uu0,so0=self.cfg.so0)

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
            self.dl = 0.0
            # data.Bph = np.sin(2*grid.TH)*0.4
            # data.Bph[0:setup.ibase,:] = 0
        if hasattr(cfg, 'so0_time_dependent'):
            self.cfg.so0 = cfg.so0_time_dependent(self.time, cfg.ett, cfg.RSUN)
            self.setup = S2MFD.Setup(self.cfg, grid)
        if hasattr(cfg, 'uu0_time_dependent'):
            self.cfg.uu0 = cfg.uu0_time_dependent(self.time, cfg.ett, cfg.RSUN)
            self.setup = S2MFD.Setup(self.cfg, grid)
        self.save()
        
    def initial_for_bisection(self, Bpht, Apht, uu0t, so0t, nt, ndt, timet, index):
        """
        Applies initial condition
        """
        cfg = self.cfg
        grid = self.grid
        setup = self.setup
        
        self.Bph = Bpht[:,:,index]
        self.Aph = Apht[:,:,index]
        self.cfg.uu0 = uu0t[index]
        self.cfg.so0 = so0t[index]
        self.n = int(nt[index])
        self.nd = int(ndt[index])
        self.time = timet[index]
        self.dl = 0.0

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
        for _ in range(20):
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
        self.cfl_condition()
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
            # if self.n % 20 == 0:
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
                if hasattr(cfg, 'so0_time_dependent'):
                    self.cfg.so0 = cfg.so0_time_dependent(self.time, cfg.ett, cfg.RSUN)
                    self.setup = S2MFD.Setup(self.cfg, grid)
                if hasattr(cfg, 'uu0_time_dependent'):
                    self.cfg.uu0 = cfg.uu0_time_dependent(self.time, cfg.ett, cfg.RSUN)
                    self.setup = S2MFD.Setup(self.cfg, grid)
                self.cfl_condition()
                print("dt=",self.dt)
                self.save()

            self.tvd_runge_kutta()


    """
    ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
    Defunction用コード
    ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
    """
    # ========================================================================================== #
    # main loop
    # TODO パラメタ変更時に設定
    parameters = {
        'a0_s': 0.0,
        'a1_s': 0.0,
        'a2_s': 0.0,
        'a3_s': 0.0,
        'b1_s': 0.0,
        'b2_s': 0.0,
        'b3_s': 0.0,
        'omega_s': 0.0,
        'a0_u': 0.0,
        'a1_u': 0.0,
        'a2_u': 0.0,
        'a3_u': 0.0,
        'b1_u': 0.0,
        'b2_u': 0.0,
        'b3_u': 0.0,
        'omega_u': 0.0,
        'u0_const': 0.0,
        's0_const': 0.0
        }
    def defunction_main_loop(self,parameters: Dict[str, float],timet,index_start,index_end):
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
        
        while self.time < timet[index_end]:
            self.time += self.dt
            self.n += 1
            if(self.time//cfg.dtout != (self.time - self.dt)//cfg.dtout):
            # if self.n % 20 == 0:
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
                
                # TODO パラメタ変更時に設定

                if hasattr(cfg, 'so0_time_dependent'):
                    parameters_s = {key: parameters[key] for key in ['a0_s', 'a1_s', 'a2_s', 'b1_s', 'b2_s', 'omega_s']}
                    self.cfg.so0 = cfg.so0_time_dependent(**parameters_s,time=self.time)
                    self.setup = S2MFD.Setup(self.cfg, grid)
                
                if hasattr(cfg, 'uu0_time_dependent'):
                    parameters_u = {key: parameters[key] for key in ['a0_u', 'a1_u', 'a2_u', 'b1_u', 'b2_u', 'omega_u']}
                    self.cfg.uu0 = cfg.uu0_time_dependent(**parameters_u,time=self.time)
                    self.setup = S2MFD.Setup(self.cfg, grid)
                    
                self.cfl_condition()
                # print("dt=",self.dt)
                self.SN[self.nd-(index_start)] = self.snumbers_energy(Bpht=self.Bph)
                self.save()

            self.tvd_runge_kutta()
    # ========================================================================================== #
    # 初期条件生成
    def initial_for_OBS(self,parameters: Dict[str, float],timet,index,index_end):
        """
        Applies initial condition
        十分なリードタイムを設ける。
        -----------------------------------------
        「内容」
        1.110年分の計算を行う（「真の初期条件」はJouve+2008のものを採用） 
        2.極小期が訪れるまで計算を継続する。
            判定方法：時間的に連続した三点の黒点数を記録、その三点のうち二番目の点が他の点より小さ
                    かったらそこを極小値とする。

        """ 
        cfg = self.cfg
        grid = self.grid
        setup = self.setup
        dir_origin = self.cfg.datadir
        self.cfg.datadir = self.cfg.datadir + "Lead_data/"
        self.initialize_simulation()

        # Lead Timeの初期条件
        self.Aph = np.zeros((grid.ixg, grid.jxg))
        self.Bph = np.zeros((grid.ixg, grid.jxg))
        self.Aph = grid.sinTH/(grid.RR/cfg.RSUN)**2*cfg.RSUN/100
        self.Aph[0:setup.ibase,:] = 0.0
        self.time = 0.0
        self.nd = 0
        self.n = 0
        self.cfg.uu0 = parameters['u0_const']
        self.cfg.so0 = parameters['s0_const']
        sn_history = np.zeros(3)

        # 110年間分計算開始
        for i in range(3):
            self.tvd_runge_kutta()
            self.time += self.dt
            sn_history[i] = self.snumbers_energy(Bpht=self.Bph)
            if(self.time//cfg.dtout != (self.time - self.dt)//cfg.dtout):
                self.save()
                self.nd += 1
                
        while self.time < 110*365*24*3600:  # 110年
            prev_Bph = self.Bph.copy()
            prev_Aph = self.Aph.copy() 
            self.tvd_runge_kutta()
            self.time += self.dt
            sn_history[0] = sn_history[1]
            sn_history[1] = sn_history[2]
            sn_history[2] = self.snumbers_energy(Bpht=self.Bph)
            if(self.time//cfg.dtout != (self.time - self.dt)//cfg.dtout):
                self.save()
                self.nd += 1
        print(f"{self.time/(365*24*3600)} [year]; 110年の計算終了")

        # 極小値が出るまで計算
        while True:
           # 極小値判定
            if (sn_history[1] < sn_history[0]) and (sn_history[1] < sn_history[2]):
                print("極小値を検出しました。計算を終了します。")
                # ここで初期条件として保存するものを整理
                self.Bpht = prev_Bph
                self.Apht = prev_Aph
                self.n    = int(0)
                self.nd   = index
                self.time = timet[index]
                self.SN = np.zeros_like(timet[index:index_end+1])
                self.SN[0] = self.snumbers_energy(Bpht=self.Bph)
                self.cfg.datadir = dir_origin    
                print(f"{self.time/(86400*365)} [year]; u0={self.cfg.uu0}; s0={self.cfg.so0}")
                self.save()
                break
            prev_Bph = self.Bph.copy()
            prev_Aph = self.Aph.copy() 
            self.tvd_runge_kutta()
            self.time += self.dt
            if(self.time//cfg.dtout != (self.time - self.dt)//cfg.dtout):
                self.save()
                self.nd += 1
            # sn_historyをシフトして新しい値を追加
            sn_history[0] = sn_history[1]
            sn_history[1] = sn_history[2]
            sn_history[2] = self.snumbers_energy(Bpht=self.Bph)
    # ========================================================================================== #

    # ========================================================================================== #
    def initial_for_OBS_prot(self,parameters: Dict[str, float],timet,index,index_end):
        """
        Applies initial condition
        十分なリードタイムを設ける。→一旦110年=1000タイムステップ
        """ 

        cfg = self.cfg
        grid = self.grid
        setup = self.setup

        # Lead Timeの初期条件
        self.Aph = np.zeros((grid.ixg, grid.jxg))
        self.Bph = np.zeros((grid.ixg, grid.jxg))
        self.Aph = grid.sinTH/(grid.RR/cfg.RSUN)**2*cfg.RSUN/100
        self.Aph[0:setup.ibase,:] = 0.0
        self.time = 0.0
        self.nd = index
        self.n = 0
        count = 0
        self.cfg.uu0 = parameters['u0_const']
        self.cfg.so0 = parameters['s0_const']
        print("[No1]","u0=",self.cfg.uu0,"time=",self.time,"nd=",self.nd)
        self.save()

        # 110年間分計算
        self.cfl_condition()
        while self.time < 110*365*24*3600:  # 110年
            self.n += 1
            self.tvd_runge_kutta()
            self.time += self.dt
            if(self.time//cfg.dtout != (self.time - self.dt)//cfg.dtout):
                self.nd += 1
                print("[No2]","u0=",self.cfg.uu0,"time=",self.time,"nd=",self.nd)
                self.save()
            if count % 2 == 0:
                SN_even = self.snumbers_energy(Bpht=self.Bph)
                tag = "even"
            else:
                SN_odds = self.snumbers_energy(Bpht=self.Bph)
                tag = "odds"
            count += 1
        prev_Bph = self.Bph.copy()
        prev_Aph = self.Aph.copy()  
        sn_history = np.zeros(3)
        self.tvd_runge_kutta()
        self.time += self.dt
        if tag == "even":
            sn_history[0] = SN_odds
            sn_history[1] = SN_even
            sn_history[2] = self.snumbers_energy(Bpht=self.Bph)
        elif tag == "odds":
            sn_history[0] = SN_even
            sn_history[1] = SN_odds
            sn_history[2] = self.snumbers_energy(Bpht=self.Bph)
        print("110年の計算終了")
        # 極小値が出るまで計算
        while True:
           # 極小値判定
            if (sn_history[1] < sn_history[0]) and (sn_history[1] < sn_history[2]):
                print("極小値を検出しました。計算を終了します。")
                self.Bpht = prev_Bph
                self.Apht = prev_Aph
                print("[No4]","u0=",self.cfg.uu0,"time=",self.time,"nd=",self.nd)
                self.save()
                break
            prev_Bph = self.Bph.copy()
            prev_Aph = self.Aph.copy() 
            self.tvd_runge_kutta()
            self.time += self.dt
            if(self.time//cfg.dtout != (self.time - self.dt)//cfg.dtout):
                self.nd += 1
                print("[No3]","u0=",self.cfg.uu0,"time=",self.time,"nd=",self.nd)
                self.save()
            # sn_historyをシフトして新しい値を追加
            sn_history[0] = sn_history[1]
            sn_history[1] = sn_history[2]
            sn_history[2] = self.snumbers_energy(Bpht=self.Bph)
    # ========================================================================================== #
    # 初期条件 
    def initial_for_defunction(self,parameters: Dict[str, float],Bpht,Apht,uu0t,so0t,nt,ndt,timet,index,index_end):
        """
        Applies initial condition
        """
        cfg = self.cfg
        grid = self.grid
        
        self.Bph = Bpht[:,:,index]
        self.Aph = Apht[:,:,index]
        self.cfg.uu0 = uu0t[index]
        self.cfg.so0 = so0t[index]
        self.time = timet[index]
        # TODO パラメタ変更時に設定
        if hasattr(cfg, 'so0_time_dependent'):
            parameters_s = {key: parameters[key] for key in ['a0_s', 'a1_s', 'a2_s', 'b1_s', 'b2_s','omega_s']}
            self.cfg.so0 = cfg.so0_time_dependent(**parameters_s,time=self.time)
            
        if hasattr(cfg, 'uu0_time_dependent'):
            parameters_u = {key: parameters[key] for key in ['a0_u', 'a1_u', 'a2_u', 'b1_u', 'b2_u','omega_u']}
            self.cfg.uu0 = cfg.uu0_time_dependent(**parameters_u,time=self.time)
            
        self.setup = S2MFD.Setup(self.cfg, grid)
        setup = self.setup
        self.n = int(nt[index])
        self.nd = int(ndt[index])
        self.SN = np.zeros_like(timet[index:index_end+1])
        self.SN[self.nd-index] = self.snumbers_energy(Bpht=self.Bph)

        self.save()
    # ========================================================================================== #
    # 判定関数①（相関係数）
    def judge(self, Sunspot_N):
        import matplotlib.pyplot as plt
        thre = 0.0
        thre = np.sqrt(np.sum((Sunspot_N - self.SN)**2))
        cc   = np.sum((Sunspot_N-np.mean(Sunspot_N))*(self.SN-np.mean(self.SN)))/np.sqrt(np.sum((Sunspot_N-np.mean(Sunspot_N))**2)*np.sum((self.SN-np.mean(self.SN))**2))
        plt.plot(Sunspot_N)
        plt.plot(self.SN)
        plt.savefig("P_sunspot.png")
        plt.clf()
        plt.close('all')
        
        print("相関係数＝",cc)
        return cc
        
    # ========================================================================================== #
    # 判定関数②（黒点総数の誤差）
    def judge2(self, Sunspot_N):
        """
        Compares the number of sunspots with the simulation results
        """
        import matplotlib.pyplot as plt
        sd = 0.0
        sd = np.sqrt((np.sum(Sunspot_N) - np.sum(self.SN))**2) / np.sum(Sunspot_N)
        print("黒点総数の誤差=",sd)
        
        return sd
    # ========================================================================================== #

    
        
    """
    ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
    以下の関数は二分法で磁場を合わせにいく関数
    ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
    """
    
    # ========================================================================================== #
    def delta1(self,x_obs,now):
        # 規格化
        upper = np.sqrt(np.sum((x_obs - now)**2*self.grid.RR*self.grid.drr*self.grid.dth))
        lower = np.sqrt(np.sum(x_obs**2*self.grid.RR*self.grid.drr*self.grid.dth))
        delta = upper / lower * 100
        return delta
    # ========================================================================================== #
    
    # ========================================================================================== #
    # 磁場を合わせにいく二分法関数（範囲指定は非可変）   
    def bisection_sources_mag(self, uu0t, Bpht, Apht):
        import matplotlib.pyplot as plt
        
        cfg = self.cfg
        grid = self.grid
        
        #　許容残差
        eps = 0.001
        obsn = self.nd
        
        
        rs = np.argmin(abs(self.grid.rr-0.6*self.cfg.RSUN))
        re = np.argmin(abs(self.grid.rr-1.0*self.cfg.RSUN))

        ts = np.argmin(abs(self.grid.th-0/180*np.pi))
        te = np.argmin(abs(self.grid.th-180/180*np.pi))
        kkk = 0

        while self.time < cfg.tend:
            # 初期値
            smin = 30
            smax = 70
            obsn = self.nd+1
            B_obs = Bpht[:,:,obsn]
            A_obs = Apht[:,:,obsn]
            def delta2(x_obs,now):
                # 規格化
                upper = np.sum((x_obs - now)*self.grid.RR[rs:re,ts:te]*self.grid.drr*self.grid.dth)
                lower = np.sqrt(np.sum(x_obs**2*self.grid.RR[rs:re,ts:te]*self.grid.drr*self.grid.dth))
                delta = upper / lower
                return delta
            breakpoint = 0
            if kkk == 1:
                break
            while True:
                smid = (smin + smax)/2
                
                # aの計算
                Bph_dfa = self.Bph
                Aph_dfa = self.Aph
                self.cfg.so0 = smin
                self.setup = S2MFD.Setup(self.cfg, grid)
                Bph_dfa, Aph_dfa = self.tvd_runge_kutta_bisection(Bph_dfa, Aph_dfa)
                aaa = delta2(A_obs[rs:re,ts:te],Aph_dfa[rs:re,ts:te])
                print("全体磁場誤差a",aaa)
                
                # bの計算
                Bph_dfb = self.Bph
                Aph_dfb = self.Aph
                self.cfg.so0 = smax
                self.setup = S2MFD.Setup(self.cfg, grid)
                Bph_dfb, Aph_dfb = self.tvd_runge_kutta_bisection(Bph_dfb, Aph_dfb)
                aab = delta2(A_obs[rs:re,ts:te],Aph_dfb[rs:re,ts:te])
                print("全体磁場誤差b",aab)
                
                # cの計算
                Bph_dfc = self.Bph
                Aph_dfc = self.Aph
                self.cfg.so0 = smid
                self.setup = S2MFD.Setup(self.cfg, grid)
                Bph_dfc, Aph_dfc = self.tvd_runge_kutta_bisection(Bph_dfc, Aph_dfc)
                aac = delta2(A_obs[rs:re,ts:te],Aph_dfc[rs:re,ts:te])
                print("全体磁場誤差c",aac)
                
                print(smax,smid,smin)
                if aac * aab < 0:
                    smin = smid
                else:
                    smax = smid
                if abs(aac) < eps:
                    self.Bph = Bph_dfc
                    self.Aph = Aph_dfc
                    self.cfg.so0 = smid
                    self.setup = S2MFD.Setup(self.cfg, grid)
                    self.time += 20*self.dt
                    self.nd += 1
                    self.n += 20
                    self.save()
                    print("=================================")
                    break
                if breakpoint < 50:
                    breakpoint += 1
                else:   
                    print("break")
                    kkk = 1
                    break
    # ========================================================================================== #        
    # ========================================================================================== #
    # 磁場を合わせにいく二分法関数（範囲指定は非可変）   
    def bisection_sources_SN(self, Sunspot_N, Bpht, Apht):
        import matplotlib.pyplot as plt
        
        cfg = self.cfg
        grid = self.grid
        
        #　許容残差
        eps = 0.00001
        obsn = self.nd
        kkk = 0


        while self.time < cfg.tend:
            smin = 30
            smax = 70
            obsn = self.nd+1
            obs = Sunspot_N[obsn]
            B_obs = Bpht[:,:,obsn]
            A_obs = Apht[:,:,obsn]
            print("目標＝",obs)
            
            def delta(now):
                return obs - now

            break_p = 0
            if kkk == 1:
                break
            
            while True:
                smid = (smin + smax)/2
                
                # aの計算
                Bph_dfa = self.Bph
                Aph_dfa = self.Aph
                self.cfg.so0 = smin
                self.setup = S2MFD.Setup(self.cfg, grid)
                Bph_dfa, Aph_dfa = self.tvd_runge_kutta_bisection(Bph_dfa, Aph_dfa)
                a_sim = self.snumbers_energy(Bph_dfa)
                aaa = delta(a_sim)
                print("黒点誤差a",aaa)
                print("磁場誤差a",self.delta1(B_obs,Bph_dfa))
                
                # bの計算
                Bph_dfb = self.Bph
                Aph_dfb = self.Aph
                self.cfg.so0 = smax
                self.setup = S2MFD.Setup(self.cfg, grid)
                Bph_dfb, Aph_dfb = self.tvd_runge_kutta_bisection(Bph_dfb, Aph_dfb)
                b_sim = self.snumbers_energy(Bph_dfb)
                aab = delta(b_sim)
                print("黒点誤差b",aab)
                print("磁場誤差b",self.delta1(B_obs,Bph_dfb))
                
                # cの計算
                Bph_dfc = self.Bph
                Aph_dfc = self.Aph
                self.cfg.so0 = smid
                self.setup = S2MFD.Setup(self.cfg, grid)
                Bph_dfc, Aph_dfc = self.tvd_runge_kutta_bisection(Bph_dfc, Aph_dfc)
                c_sim = self.snumbers_energy(Bph_dfc)
                aac = delta(c_sim)
                print("黒点誤差c",aac)
                print("磁場誤差c",self.delta1(B_obs,Bph_dfc))
                
                print(smax,smid,smin)
                if delta(a_sim) * delta(c_sim) < 0:
                    smax = smid
                else:
                    smin = smid
                if abs(delta(c_sim)) < eps:
                    self.Bph = Bph_dfc
                    self.Aph = Aph_dfc
                    self.cfg.so0 = smid
                    self.setup = S2MFD.Setup(self.cfg, grid)
                    self.time += 20*self.dt
                    self.nd += 1
                    self.n += 20
                    self.dl = self.delta1(B_obs,Bph_dfc)
                    self.save()
                    print("=================================")
                    break
                if break_p < 100:
                    break_p += 1
                else:
                    print("break")
                    kkk = 1
                    break
    # ========================================================================================== # 

    # ========================================================================================== #
    # 磁場を合わせにいく二分法関数（範囲指定は非可変）   
    def bisection_meridional_mag_1(self, uu0t, Bpht, Apht):
        import matplotlib.pyplot as plt
        
        cfg = self.cfg
        grid = self.grid
        
        #　許容残差
        eps = 0.001
        obsn = self.nd
        
        
        rs = np.argmin(abs(self.grid.rr-(0.7-0.05)*self.cfg.RSUN))
        re = np.argmin(abs(self.grid.rr-(0.7+0.05)*self.cfg.RSUN))

        ts = np.argmin(abs(self.grid.th-50/180*np.pi))
        te = np.argmin(abs(self.grid.th-75/180*np.pi))
        kkk = 0

        while self.time < cfg.tend:
            # 初期値
            umin = 400
            umax = 1600
            obsn = self.nd+1
            B_obs = Bpht[:,:,obsn]
            A_obs = Apht[:,:,obsn]
            def delta2(x_obs,now):
                # 規格化
                upper = np.sqrt(np.sum((x_obs - now)**2*self.grid.RR[rs:re,ts:te]*self.grid.drr*self.grid.dth))
                lower = np.sqrt(np.sum(x_obs**2*self.grid.RR[rs:re,ts:te]*self.grid.drr*self.grid.dth))
                delta = upper / lower * 100
                return delta
            breakpoint = 0
            if kkk == 1:
                break
            while True:
                umid = (umin + umax)/2
                
                # aの計算
                Bph_dfa = self.Bph
                Aph_dfa = self.Aph
                self.cfg.uu0 = umin
                self.setup = S2MFD.Setup(self.cfg, grid)
                Bph_dfa, Aph_dfa = self.tvd_runge_kutta_bisection(Bph_dfa, Aph_dfa)
                aaa = delta2(B_obs[rs:re,ts:te],Bph_dfa[rs:re,ts:te])
                print("範囲磁場誤差a",aaa)
                print("全体磁場誤差a",self.delta1(B_obs,Bph_dfa))
                
                # bの計算
                Bph_dfb = self.Bph
                Aph_dfb = self.Aph
                self.cfg.uu0 = umax
                self.setup = S2MFD.Setup(self.cfg, grid)
                Bph_dfb, Aph_dfb = self.tvd_runge_kutta_bisection(Bph_dfb, Aph_dfb)
                aab = delta2(B_obs[rs:re,ts:te],Bph_dfb[rs:re,ts:te])
                print("範囲磁場誤差b",aab)
                print("全体磁場誤差b",self.delta1(B_obs,Bph_dfb))
                
                # cの計算
                Bph_dfc = self.Bph
                Aph_dfc = self.Aph
                self.cfg.uu0 = umid
                self.setup = S2MFD.Setup(self.cfg, grid)
                Bph_dfc, Aph_dfc = self.tvd_runge_kutta_bisection(Bph_dfc, Aph_dfc)
                aac = delta2(B_obs[rs:re,ts:te],Bph_dfc[rs:re,ts:te])
                print("範囲磁場誤差c",aac)
                print("全体磁場誤差c",self.delta1(B_obs,Bph_dfc))
                
                print(umax,umid,umin)
                if aaa - aab < 0:
                    umax = umid
                else:
                    umin = umid
                if aac < eps:
                    self.Bph = Bph_dfc
                    self.Aph = Aph_dfc
                    self.cfg.uu0 = umid
                    self.setup = S2MFD.Setup(self.cfg, grid)
                    self.time += 20*self.dt
                    self.nd += 1
                    self.n += 20
                    self.dl = self.delta1(B_obs,Bph_dfc)
                    self.save()
                    print("=================================")
                    break
                if breakpoint < 100:
                    breakpoint += 1
                else:   
                    print("break")
                    kkk = 1
                    break
    # ========================================================================================== #
    
    # ========================================================================================== #
    # 磁場を合わせにいく二分法関数（範囲指定は可変）       
    def bisection_meridional_mag_2(self, uu0t, Bpht, Apht, rr_manege, th_manege, ii, jj):
        import matplotlib.pyplot as plt
        
        cfg = self.cfg
        grid = self.grid
        
        #　許容残差
        eps = 0.001
        obsn = self.nd
        
        
        rs = np.argmin(abs(self.grid.rr-rr_manege[0,ii]*self.cfg.RSUN))
        re = np.argmin(abs(self.grid.rr-rr_manege[1,ii]*self.cfg.RSUN))

        ts = np.argmin(abs(self.grid.th-th_manege[0,jj]/180*np.pi))
        te = np.argmin(abs(self.grid.th-th_manege[1,jj]/180*np.pi))
        kkk = 0

        while self.time < cfg.tend:
            # 初期値
            umin = 400
            umax = 1600
            obsn = self.nd+1
            B_obs = Bpht[:,:,obsn]
            A_obs = Apht[:,:,obsn]
            def delta2(x_obs,now):
                # 規格化
                upper = np.sqrt(np.sum((x_obs - now)**2*self.grid.RR[rs:re,ts:te]*self.grid.drr*self.grid.dth))
                lower = np.sqrt(np.sum(x_obs**2*self.grid.RR[rs:re,ts:te]*self.grid.drr*self.grid.dth))
                delta = upper / lower * 100
                return delta
            breakpoint = 0
            if kkk == 1:
                break
            while True:
                umid = (umin + umax)/2
                
                # aの計算
                Bph_dfa = self.Bph
                Aph_dfa = self.Aph
                self.cfg.uu0 = umin
                self.setup = S2MFD.Setup(self.cfg, grid)
                Bph_dfa, Aph_dfa = self.tvd_runge_kutta_bisection(Bph_dfa, Aph_dfa)
                aaa = delta2(B_obs[rs:re,ts:te],Bph_dfa[rs:re,ts:te])
                print("全体磁場誤差a",aaa)
                
                # bの計算
                Bph_dfb = self.Bph
                Aph_dfb = self.Aph
                self.cfg.uu0 = umax
                self.setup = S2MFD.Setup(self.cfg, grid)
                Bph_dfb, Aph_dfb = self.tvd_runge_kutta_bisection(Bph_dfb, Aph_dfb)
                aab = delta2(B_obs[rs:re,ts:te],Bph_dfb[rs:re,ts:te])
                print("全体磁場誤差b",aab)
                
                # cの計算
                Bph_dfc = self.Bph
                Aph_dfc = self.Aph
                self.cfg.uu0 = umid
                self.setup = S2MFD.Setup(self.cfg, grid)
                Bph_dfc, Aph_dfc = self.tvd_runge_kutta_bisection(Bph_dfc, Aph_dfc)
                aac = delta2(B_obs[rs:re,ts:te],Bph_dfc[rs:re,ts:te])
                print("全体磁場誤差c",aac)
                
                print(umax,umid,umin)
                if aaa - aab < 0:
                    umax = umid
                else:
                    umin = umid
                if aac < eps:
                    self.Bph = Bph_dfc
                    self.Aph = Aph_dfc
                    self.cfg.uu0 = umid
                    self.setup = S2MFD.Setup(self.cfg, grid)
                    self.time += 20*self.dt
                    self.nd += 1
                    self.n += 20
                    self.save()
                    print("=================================")
                    break
                if breakpoint < 100:
                    breakpoint += 1
                else:   
                    print("break")
                    kkk = 1
                    break
    # ========================================================================================== #
    
    # ========================================================================================== #
    # main_loop(黒点数)
    def main_loop_for_bisection_prot(self, Sunspot_N, Bpht, Apht):
        import matplotlib.pyplot as plt
        
        cfg = self.cfg
        grid = self.grid
        
        #　許容残差
        eps = 0.00001
        obsn = self.nd
        kkk = 0


        while self.time < cfg.tend:
            # 初期値
            umin = 0
            umax = 2000
            obsn = self.nd+1
            obs = Sunspot_N[obsn]
            B_obs = Bpht[:,:,obsn]
            A_obs = Apht[:,:,obsn]
            print("目標＝",obs)
            
            def delta(now):
                return obs - now

            break_p = 0
            if kkk == 1:
                break
            
            while True:
                umid = (umin + umax)/2
                
                ########消す##########
                """
                uu0 = np.linspace(400,1600,100)
                a_sim = np.zeros(100)
                for ie in range(100):
                    Bph_df = self.Bph
                    Aph_df = self.Aph
                    self.cfg.uu0 = uu0[ie]
                    self.setup = S2MFD.Setup(self.cfg, grid)
                    Bph_df, Aph_df = self.tvd_runge_kutta_bisection(Bph_df, Aph_df)
                    a_sim[ie] = self.snumbers_energy(Bph_df)
                    aaa = delta(a_sim[ie])
                    print("黒点誤差a",aaa)
                    print("黒点数",a_sim[ie])
                plt.clf()
                plt.scatter(uu0,a_sim,s=10)
                plt.savefig("meridional_flow_speed_al.png")
                plt.clf()
                print(obsn,'step')
                print(np.max(a_sim)-np.min(a_sim))
                break
                """
                ########消す##########
                
                # aの計算
                Bph_dfa = self.Bph
                Aph_dfa = self.Aph
                self.cfg.uu0 = umin
                self.setup = S2MFD.Setup(self.cfg, grid)
                Bph_dfa, Aph_dfa = self.tvd_runge_kutta_bisection(Bph_dfa, Aph_dfa)
                a_sim = self.snumbers_energy(Bph_dfa)
                aaa = delta(a_sim)
                print("黒点誤差a",aaa)
                print("磁場誤差a",self.delta1(B_obs,Bph_dfa))
                
                # bの計算
                Bph_dfb = self.Bph
                Aph_dfb = self.Aph
                self.cfg.uu0 = umax
                self.setup = S2MFD.Setup(self.cfg, grid)
                Bph_dfb, Aph_dfb = self.tvd_runge_kutta_bisection(Bph_dfb, Aph_dfb)
                b_sim = self.snumbers_energy(Bph_dfb)
                aab = delta(b_sim)
                print("黒点誤差b",aab)
                print("磁場誤差b",self.delta1(B_obs,Bph_dfb))
                
                # cの計算
                Bph_dfc = self.Bph
                Aph_dfc = self.Aph
                self.cfg.uu0 = umid
                self.setup = S2MFD.Setup(self.cfg, grid)
                Bph_dfc, Aph_dfc = self.tvd_runge_kutta_bisection(Bph_dfc, Aph_dfc)
                c_sim = self.snumbers_energy(Bph_dfc)
                aac = delta(c_sim)
                print("黒点誤差c",aac)
                print("磁場誤差c",self.delta1(B_obs,Bph_dfc))
                
                print(umax,umid,umin)
                if delta(a_sim) * delta(c_sim) < 0:
                    umax = umid
                else:
                    umin = umid
                if abs(delta(c_sim)) < eps:
                    self.Bph = Bph_dfc
                    self.Aph = Aph_dfc
                    self.cfg.uu0 = umid
                    self.setup = S2MFD.Setup(self.cfg, grid)
                    self.time += 20*self.dt
                    self.nd += 1
                    self.n += 20
                    self.dl = self.delta1(B_obs,Bph_dfc)
                    self.save()
                    print("=================================")
                    break
                if break_p < 100:
                    break_p += 1
                else:
                    print("break")
                    kkk = 1
                    break
            ########消す##########
            """
            break
            """
            ########消す##########
                
    # ========================================================================================== #

    # ========================================================================================== #
    # main_loop(黒点数)
    def main_loop_for_bisection(self, Sunspot_N, Bpht, Apht):
        import matplotlib.pyplot as plt
        
        cfg = self.cfg
        grid = self.grid
        
        #　許容残差
        eps = 0.00001
        obsn = self.nd
        kkk = 0


        while self.time < cfg.tend:
            # 初期値
            obsn = self.nd+1
            obs = Sunspot_N[obsn]
            B_obs = Bpht[:,:,obsn]
            A_obs = Apht[:,:,obsn]
            umin = 400
            umax = 1600
            ka  = 0
            # print('umin=',umin,'umax=',umax)
            
            def delta(now):
                return obs - now

            break_p = 0
            if kkk == 1:
                break
            
            while True:
                umid = (umin + umax)/2
                
                # aの計算
                Bph_dfa = self.Bph
                Aph_dfa = self.Aph
                self.cfg.uu0 = umin
                self.setup = S2MFD.Setup(self.cfg, grid)
                Bph_dfa, Aph_dfa = self.tvd_runge_kutta_bisection(Bph_dfa, Aph_dfa)
                a_sim = self.snumbers_energy(Bph_dfa)
                aaa = delta(a_sim)
                print("黒点誤差a",aaa)
                print("磁場誤差a",self.delta1(B_obs,Bph_dfa))
                
                # bの計算
                Bph_dfb = self.Bph
                Aph_dfb = self.Aph
                self.cfg.uu0 = umax
                self.setup = S2MFD.Setup(self.cfg, grid)
                Bph_dfb, Aph_dfb = self.tvd_runge_kutta_bisection(Bph_dfb, Aph_dfb)
                b_sim = self.snumbers_energy(Bph_dfb)
                aab = delta(b_sim)
                print("黒点誤差b",aab)
                print("磁場誤差b",self.delta1(B_obs,Bph_dfb))
                
                # cの計算
                Bph_dfc = self.Bph
                Aph_dfc = self.Aph
                self.cfg.uu0 = umid
                self.setup = S2MFD.Setup(self.cfg, grid)
                Bph_dfc, Aph_dfc = self.tvd_runge_kutta_bisection(Bph_dfc, Aph_dfc)
                c_sim = self.snumbers_energy(Bph_dfc)
                aac = delta(c_sim)
                print("黒点誤差c",aac)
                print("磁場誤差c",self.delta1(B_obs,Bph_dfc))
                
                print(umax,umid,umin)
                
                for im in range(2,15):
                    if Sunspot_N[obsn-im]>Sunspot_N[obsn-(im+1)] and Sunspot_N[obsn-im]>Sunspot_N[obsn-(im-1)]:
                        Bph_df = self.Bph
                        Aph_df = self.Aph
                        startpoint = 500
                        uu0t = self.maximum_bisection(datadir="data",startpoint=startpoint)
                        self.cfg.uu0 = 2*uu0t[obsn-startpoint-1]-uu0t[obsn-startpoint-2]
                        self.setup = S2MFD.Setup(self.cfg, grid)
                        Bph_df, Aph_df = self.tvd_runge_kutta_bisection(Bph_df, Aph_df)
                        sim = self.snumbers_energy(Bph_df)
                        aa = delta(sim)
                        print("黒点誤差(特例)",aa)
                        print("磁場誤差（特例）",self.delta1(B_obs,Bph_df))
                        self.Bph = Bph_df
                        self.Aph = Aph_df
                        self.time += 20*self.dt
                        self.nd += 1
                        self.n += 20
                        self.dl = self.delta1(B_obs,Bph_df)
                        self.save()
                        ka=1
                        print("=================================")
                        break
                
                if delta(a_sim) * delta(c_sim) < 0:
                    umax = umid
                else:
                    umin = umid
                if abs(delta(c_sim)) < eps or break_p == 75:
                    self.Bph = Bph_dfc
                    self.Aph = Aph_dfc
                    self.cfg.uu0 = umid
                    self.setup = S2MFD.Setup(self.cfg, grid)
                    self.time += 20*self.dt
                    self.nd += 1
                    self.n += 20
                    self.dl = self.delta1(B_obs,Bph_dfc)
                    self.save()
                    print("=================================")
                    break
                if break_p < 50:
                    break_p += 1
                if ka == 1:
                    break
                
    # ========================================================================================== #
    
    # ========================================================================================== #
    # 二分法シミュレーションのための黒点数を数えておくコード（磁気エネルギー）
    def snumbers_energy(self,Bpht):
        base  = 1+np.argmin(abs(self.grid.rr-0.7*self.cfg.RSUN))
        base_1= 1+np.argmin(abs(self.grid.rr-(0.7-0.05)*self.cfg.RSUN))
        base_2= 1+np.argmin(abs(self.grid.rr-(0.7+0.05)*self.cfg.RSUN))
        loca  =   np.argmin(abs(self.grid.th- 75/180*np.pi))
        locap =   np.argmin(abs(self.grid.th- 50/180*np.pi))
        locam =   np.argmin(abs(self.grid.th- 130/180*np.pi))
        
        
        SN = Bpht[base,loca]**2
        # SN = np.mean(Bpht[base_1:base_2,locap:locam]**2,axis=(0,1))
        
        # n_conv = 4 #移動平均の個数
        # conv_f = np.ones(n_conv)/n_conv
        # SN2 = np.convolve(SN, conv_f, mode='same')#移動平均
        
        return SN
    # ========================================================================================== # 
    
    # ========================================================================================== #
    # 二分法シミュレーションのための黒点数を数えておくコード（磁気エネルギー）
    def pre_snumbers_energy(self,Bpht):
        base  = 1+np.argmin(abs(self.grid.rr-0.7*self.cfg.RSUN))
        base_1= 1+np.argmin(abs(self.grid.rr-(0.7-0.05)*self.cfg.RSUN))
        base_2= 1+np.argmin(abs(self.grid.rr-(0.7+0.05)*self.cfg.RSUN))
        loca  =   np.argmin(abs(self.grid.th- 75/180*np.pi))
        locap =   np.argmin(abs(self.grid.th- 50/180*np.pi))
        locam =   np.argmin(abs(self.grid.th- 130/180*np.pi))
        
        
        SN = Bpht[base,loca,:]**2
        # SN = np.mean(Bpht[base_1:base_2,locap:locam,:]**2,axis=(0,1))
        
        n_conv = 4 #移動平均の個数
        conv_f = np.ones(n_conv)/n_conv
        SN2 = np.convolve(SN, conv_f, mode='same')#移動平均
        
        return SN,SN2
    # ========================================================================================== # 
    
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

    # ========================================================================================== #
    # 二分法シミュレーションのためのデータを読み込む関数  
    def load_for_bisection(self,datadir):
        import os
        datadir = datadir+'/'
        data = S2MFD.Data.initial_load(datadir)

        cfg = data.cfg
        grid = data.grid
        setup = data.setup

        n1 = 0
        if os.path.isdir(datadir):
            # dataディレクトリ内の最も大きな番号を探る
            # 特定のステップから始めたい場合は、そのステップを手で指定する
            files = os.listdir(datadir)
            for file in files:
                filel = file.split('.')
                if filel[0] == 'data':
                    n1 = max(n1, int(filel[1]))

    #  最後まで読み取れていなかったので一つ追加
        n1=n1+1
        
        n0 = 0
        tau_diff = data.cfg.RSUN**2/data.cfg.ett
        timet = np.zeros(n1-n0)
        nt = np.zeros(n1-n0)
        ndt = np.zeros(n1-n0)
        Brrt = np.zeros((grid.ixg,grid.jxg,n1-n0))
        Btht = np.zeros((grid.ixg,grid.jxg,n1-n0))
        Bpht = np.zeros((grid.ixg,grid.jxg,n1-n0))
        Apht = np.zeros((grid.ixg,grid.jxg,n1-n0))
        so0t = np.zeros(n1-n0)
        uu0t = np.zeros(n1-n0)

        for n  in range(n0,n1):
            print(n)
            data.data_load(n)
            Brr, Bth = S2MFD.physics.poloidal_mag(data.Aph, grid.RR, grid.sinTH, grid.drr, grid.dth)
            d = np.load(file=datadir+'data.'+str(n).zfill(6)+'.npz')
            timet[n-n0] = d['time']
            nt[n-n0] = d['n']
            ndt[n-n0] = d['nd']
            Brrt[:,:,n-n0] = Brr
            Btht[:,:,n-n0] = Bth
            Bpht[:,:,n-n0] = d['Bph']
            Apht[:,:,n-n0] = d['Aph']
            so0t[n-n0] = d['so0']
            uu0t[n-n0] = d['uu0']
            
        return n1, Bpht, Apht, uu0t, so0t, nt, ndt, timet
    # ========================================================================================== #


    # ========================================================================================== #
    # 極大期後のためのデータ読み込み 
    def maximum_bisection(self,datadir,startpoint):
        import os
        datadir = datadir+'/'
        data = S2MFD.Data.initial_load(datadir)

        cfg = data.cfg
        grid = data.grid
        setup = data.setup

        n1 = 0
        if os.path.isdir(datadir):
            # dataディレクトリ内の最も大きな番号を探る
            # 特定のステップから始めたい場合は、そのステップを手で指定する
            files = os.listdir(datadir)
            for file in files:
                filel = file.split('.')
                if filel[0] == 'data':
                    n1 = max(n1, int(filel[1]))

        n1 = n1+1
        n0 = startpoint
        tau_diff = data.cfg.RSUN**2/data.cfg.ett
        timet = np.zeros(n1-n0)
        nt = np.zeros(n1-n0)
        ndt = np.zeros(n1-n0)
        Brrt = np.zeros((grid.ixg,grid.jxg,n1-n0))
        Btht = np.zeros((grid.ixg,grid.jxg,n1-n0))
        Bpht = np.zeros((grid.ixg,grid.jxg,n1-n0))
        Apht = np.zeros((grid.ixg,grid.jxg,n1-n0))
        so0t = np.zeros(n1-n0)
        uu0t = np.zeros(n1-n0)

        for n  in range(n0,n1):
            print(n)
            data.data_load(n)
            Brr, Bth = S2MFD.physics.poloidal_mag(data.Aph, grid.RR, grid.sinTH, grid.drr, grid.dth)
            d = np.load(file=datadir+'data.'+str(n).zfill(6)+'.npz')
            timet[n-n0] = d['time']
            nt[n-n0] = d['n']
            ndt[n-n0] = d['nd']
            Brrt[:,:,n-n0] = Brr
            Btht[:,:,n-n0] = Bth
            Bpht[:,:,n-n0] = d['Bph']
            Apht[:,:,n-n0] = d['Aph']
            so0t[n-n0] = d['so0']
            uu0t[n-n0] = d['uu0']
            
        return uu0t
    # ========================================================================================== #


__all__ = [
         'initialize',
         'cfl_condition',
         'save',
         'initial_condition',
         'tvd_runge_kutta',
         'main_loop',
         ]