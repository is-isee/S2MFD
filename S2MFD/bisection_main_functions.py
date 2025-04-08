sys.path.append('../')
import S2MFD

# ========================================================================================== #
# 二分法シミュレーションコード                          
def bisection_simulation(cfg=None, parameter_file=None, datadir=None):
   if datadir is None:
      print('You need to specify the datadir')
      return   
   cfg, grid, n1, Bpht, uu0t = load_for_bisection(datadir)
   Sunspot_N = snumbers_for_bisection(cfg, grid, n1, Bpht)
   if parameter_file is None:
      cfg = S2MFD.Cfg()
   else:
      cfg = S2MFD.Cfg(parameter_file)
   cfg.datadir = 'data/'
   sim = S2MFD.Simulation(cfg)
   sim.initialize_simulation()
   sim.cfl_condition()
   sim.initial_for_bisection()
   sim.main_loop_for_bisection(Sunspot_N=Sunspot_N,uu0t=uu0t)
# ========================================================================================== #

# ========================================================================================== #
# 二分法シミュレーションのためのデータを読み込む関数  
def load_for_bisection(datadir):
    datadir = datadir+'/'
    data = S2MFD.Data.initial_load(datadir)

    cfg = data.cfg
    grid = data.grid
    setup = data.setup

    fig = plt.figure('dynamo',figsize=(10,10))

    n1 = 0
    if os.path.isdir(datadir):
        # dataディレクトリ内の最も大きな番号を探る
        # 特定のステップから始めたい場合は、そのステップを手で指定する
        files = os.listdir(datadir)
        for file in files:
            filel = file.split('.')
            if filel[0] == 'data':
                n1 = max(n1, int(filel[1]))

    n0 = 0
    tau_diff = data.cfg.RSUN**2/data.cfg.ett
    timet = np.zeros(n1-n0)
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
        Brrt[:,:,n-n0] = Brr
        Btht[:,:,n-n0] = Bth
        Bpht[:,:,n-n0] = d['Bph']
        Apht[:,:,n-n0] = d['Aph']
        so0t[n-n0] = d['so0']
        uu0t[n-n0] = d['uu0']
        
    return cfg, grid, n1, Bpht, uu0t
# ========================================================================================== #

# ========================================================================================== #
# 二分法シミュレーションのための黒点数を数えておくコード
def snumbers_for_bisection(cfg,grid,n1,Bpht):
   # 黒点数の計上
   N_num = np.zeros(n1)
   S_num = np.zeros(n1)
   thrsh = 3.0
   base  = 1+np.argmin(abs(grid.rr-0.7*cfg.RSUN))
   stat  = 1+np.argmin(abs(grid.th- 40/180*np.pi))
   endd  = 1+np.argmin(abs(grid.th-140/180*np.pi))
   # グリッドに応じた北半球と南半球の分割
   if grid.jxg % 2==0: 
      S_equa = grid.jxg//2 - 1
      N_equa = S_equa + 1
   else:
      S_equa = (grid.jxg-1)//2 - 1
      N_equa = S_equa + 2

   # S_numとN_numは全く同じになる（南北対称だから当たり前）
   # n1は時間要素の最後の番号
   for t in range(n1):
      # S_num[t] = np.sum(Bpht[base,grid.margin:S_equa+1,t]<-thrsh) + np.sum(Bpht[base,grid.margin:S_equa+1,t]>thrsh)
      # N_num[t] = np.sum(Bpht[base,N_equa:grid.jxg-grid.margin,t]<-thrsh) + np.sum(Bpht[base,N_equa:grid.jxg-grid.margin,t]>thrsh)
      S_num[t] = np.sum(Bpht[base,stat:S_equa+1,t]<-thrsh) + np.sum(Bpht[base,stat:S_equa+1,t]>thrsh)
      N_num[t] = np.sum(Bpht[base,N_equa:endd,t]<-thrsh)   + np.sum(Bpht[base,N_equa:endd,t]>thrsh)
   # 観測に基づいた調整パラメタ
   kappa = 0.3
   S_num = kappa * S_num
   N_num = kappa * N_num

   n_conv = 4 #移動平均の個数
   conv_f = np.ones(n_conv)/n_conv
   S_num2 = np.convolve(S_num, conv_f, mode='same')#移動平均
   N_num2 = np.convolve(N_num, conv_f, mode='same')#移動平均
   
   return S_num
# ========================================================================================== #