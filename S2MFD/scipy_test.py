sys.path.append('../')
import S2MFD

# scipy提供のルジャンドル陪関数
import scipy.special
def scipy_le():
    le = np.zeros((100,len(grid.cosTH)))
    for i in range(0,100):
        le[i,:] = scipy.special.lpmv(1,i,grid.cosTH[0,:])
    # plt.plot(grid.cosTH[0,:],le[:,20])
    return le

# 直交性の検証
def orthogonality(n):
    ale = np.zeros(legendre.lmax)
    sinth = np.sin(grid.th[grid.margin:grid.jxg-grid.margin])
    for m in range(0,legendre.lmax): 
        for i in range(0, grid.jx):
            ale[m] += legendre.P1n[i,n]*legendre.P1n[i,m]*sinth[i]*grid.dth
    return ale
# Dikpati+1994(33)式の検証
def test_33():
    # Rでのベクトルポテンシャル
    AR = np.linspace(0,grid.jx,128)
    ARe = np.zeros(grid.jx)
    
    # a_n(t)の項、n個あるのでルジャンドル陪関数のnに従う
    ant = np.zeros(legendre.lmax)
    
    # 0<θ<πで定義
    sinth = np.sin(grid.th[grid.margin:grid.jxg-grid.margin])
    
    # (34)式の積分、nに従う
    itg = np.zeros(legendre.lmax)
    
    # cfg.RSUNを使うと数字が大きすぎて無理
    RSUN = cfg.RSUN/cfg.RSUN
    
    # a_n(t)
    # TODO n=0は定義できない。
    for n in range(1, legendre.lmax):
        # integrate (range:0~π)
        for j in range(0, grid.jx):
            itg[n] += AR[j]*legendre.P1n[n,j]*sinth[j]* grid.dth
        # a_n(t)
        ant[n] = (2*n+1)*RSUN**(n+1)/(2*n*(n+1))*itg[n]
        # 1~nまで足しあげる（Σ）
        ARe += ant[n]/RSUN**(n+1)*legendre.P1n[n]
    return AR, ARe, itg, ant

# Dikpati+1994(33)式の検証
def test_33_an(nn):
    # Rでのベクトルポテンシャル（太陽内部）
    # AR = np.random.rand(grid.jx)
    itg=0
    RSUN = 2.0
    AR = np.zeros(grid.jx)
    # a_n(t)の項、n個あるのでルジャンドル陪関数のnに従う
    ant = np.random.randn(legendre.lmax)
    
    # 0<θ<πで定義
    sinth = np.sin(grid.th[grid.margin:grid.jxg-grid.margin])
    AR = ant[nn]/(RSUN**(nn+1)) * legendre.P1n[nn]
    for j in range(0, grid.jx):
        itg += AR[j]*legendre.P1n[nn,j]*sinth[j]* grid.dth
        # a_n(t)
    ant_2 = (2*nn+1)*(RSUN**(nn+1)) / (2*nn*(nn+1))*itg
        
    return AR, itg, ant, ant_2

def test_33_ans(nn):
    # Σではうまくいかない
    AR    = np.zeros(grid.jx)
    ant   = np.random.rand(legendre.lmax)
    ant_2 = 0.0
    itg   = 0.0
    
    sinth = np.sin(grid.th[grid.margin:grid.jxg-grid.margin])
    RSUN  = 2
    for n in range(1,legendre.lmax):
        AR += ant[n] / RSUN**(n+1) *  legendre.P1n[n]
    for j in range(0, grid.jx):
        itg += AR[j]*legendre.P1n[nn,j]*sinth[j]* grid.dth
        # a_n(t)
    ant_2 = (2*nn+1)*(RSUN**(nn+1)) / (2*nn*(nn+1))*itg
    
    return AR, itg, ant, ant_2

# 直交性は満たすものとする
def test_33_ann():
    AR    = np.zeros((legendre.lmax,grid.jx))
    ant   = np.random.rand(legendre.lmax)
    ant_2 = np.zeros_like(ant)
    itg   = np.zeros_like(ant)
    
    sinth = np.sin(grid.th[grid.margin:grid.jxg-grid.margin])
    RSUN  = 2
    for n in range(1,legendre.lmax):
        AR[n] = ant[n] / RSUN**(n+1) *  legendre.P1n[n]
        for j in range(0, grid.jx):
            itg[n] += AR[n,j]*legendre.P1n[n,j]*sinth[j]* grid.dth
        # a_n(t)
        ant_2[n] = (2*n+1)*(RSUN**(n+1)) / (2*n*(n+1))*itg[n]
    
    return AR, itg, ant, ant_2


def potential_test(Aph,add,types):
    ant = np.zeros(legendre.lmax-1) # (127(n),)
    sinth = np.sin(grid.th[grid.margin:grid.jxg - grid.margin])  # θのサイン値(128(θ),)
    n_values = np.arange(1, legendre.lmax)  # n のインデックスを作成(1~127)
    P1Sum = np.zeros_like(sinth)  # 合計用配列(128(θ),)
    
    Aph_reduced = Aph[grid.ixg - 2, 1:grid.jx+1]  # 太陽表面のAφ(128(θ),)、1~128
    P1n_reduced = legendre.P1n[1:, :]   # (127(n), 128(θ))
    
    n_values_ex, sinth_ex  = np.meshgrid(n_values, sinth, indexing = 'ij') # (127(n), 128(θ))
    n_values_ex, Aph_ex    = np.meshgrid(n_values, Aph_reduced,indexing = 'ij') # (127(n), 128(θ))
    
    # 比較用
    Aph_num = np.zeros(grid.jx)
    Aph_ana = np.zeros(grid.jx)
    Aph_rad = np.zeros(grid.jx)
    
    # 外側の準備
    Aph_expand = np.zeros((grid.ix+add,grid.jx))
    rr_ex = np.zeros(grid.ix+add)
    th_ex = grid.th[1:grid.jx+1]
    for n in range(grid.ix):
        rr_ex[n] = grid.rr[n]
    for n in range(grid.ix,grid.ix + add):
        rr_ex[n] = grid.rr[grid.ix] + (n - grid.ix)*grid.drr
    RR_ex, TH_ex = np.meshgrid(rr_ex, th_ex, indexing='ij')
    X_expand, Y_expand = RR_ex * np.cos(TH_ex), RR_ex * np.sin(TH_ex)
    
    coefficients = (2 * n_values + 1) / (2 * n_values * (n_values + 1)) # (127(n),)
    # 0~πの積分
    itg = np.sum(Aph_ex * P1n_reduced * sinth_ex * grid.dth, axis=1) # (127(n),)
    # ant の計算 (ベクトル化済み)
    ant = coefficients * itg # (127(n),)
    # θを追加
    ant_ex, dammy  = np.meshgrid(ant, sinth, indexing = 'ij') # (127(n), 128(θ))

    # top boundary numerical
    Aph_num = +Aph_reduced - grid.drr * np.sum((n_values_ex + 1) * ant_ex * (grid.rr[grid.ixg-grid.margin-1]/grid.rr[grid.ixg-1])**(n_values_ex + 1) * (1/grid.rr[grid.ixg-1]) * P1n_reduced,axis=0)
    
    # top boundary analytical
    Aph_ana = np.sum(ant_ex * (grid.rr[grid.ixg-2] / grid.rr[grid.ixg-1])**(n_values_ex + 1) * P1n_reduced,axis=0)
    
    # top boundary radial
    Aph_rad = +Aph_reduced / grid.rr[grid.ixg-1] * grid.rr[grid.ixg-2]
    
    # 外側のポテンシャル磁場をかく
    for i in range(grid.ix+add):
        if i < grid.ix:
            Aph_expand[i,:] = Aph[i+1,1:grid.jx+1]
        else:
            if types == "ana":
                Aph_expand[i,:] = np.sum(ant_ex * (grid.rr[grid.jxg-2]/rr_ex[i])**(n_values_ex + 1) * P1n_reduced,axis=0)
            elif types == "num":
                Aph_expand[i,:] = Aph_expand[i-1,:] - grid.drr*np.sum((n_values_ex + 1) * ant_ex * (grid.rr[grid.ixg-2]/rr_ex[i])**(n_values_ex + 1) * (1/rr_ex[i]) * P1n_reduced,axis=0)
    
    return Aph_ana, Aph_num, Aph_expand, X_expand, Y_expand, RR_ex, TH_ex

# 描画用コード
def make_medi(Aph):
    plt.clf()
    plt.close('all')
    fig = plt.figure('dynamo',figsize=(5,10))   
    ax = fig.add_subplot(111,aspect='equal')
    ax.clear()
    ax.contour(Y_expand/cfg.RSUN,X_expand/cfg.RSUN,RR_ex/cfg.RSUN*np.sin(TH_ex)*Aph/cfg.RSUN,colors='black',levels=np.linspace(-0.02,0.02,16))
    radius = grid.rrmax/cfg.RSUN
    ax.plot(radius*np.sin(grid.th),radius*np.cos(grid.th),color='black',alpha=0.4)
    radius = grid.rrmin/cfg.RSUN         
    ax.plot(radius*np.sin(grid.th),radius*np.cos(grid.th),color='black',alpha=0.4)
    # ax.set_xlim( 0,1)
    # ax.set_ylim(-1,1)

# ナイキスト波数の検討
# Nyquistで第何項まで書くか指定
def test_33_nyquist(Nyquist):
    # 0<θ<πで定義
    costh = np.cos(grid.th[grid.margin:grid.jxg-grid.margin])
    lmax = Nyquist
    P1n = np.zeros((lmax,grid.jx))
    # n = 1
    P1n[1,:] = -(1-costh**2)**0.5
    # n = 2, 3, ...
    for i in range(2, lmax):
        # recurrence relation
        P1n[i,:] = ((2*i-1)/(i-1))*costh*P1n[i-1,:]-(i/(i-1))*P1n[i-2,:]
    # Rでのベクトルポテンシャル（太陽内部）
    # AR = np.random.rand(grid.jx)
    AR = np.zeros(grid.jx)
    # a_n(t)の項、n個あるのでルジャンドル陪関数のnに従う
    ant = np.random.rand(lmax)
    ant_2 = np.zeros(lmax)
    
    # 0<θ<πで定義
    sinth = np.sin(grid.th[grid.margin:grid.jxg-grid.margin])
    
    # (34)式の積分、nに従う
    itg = np.zeros(lmax)
    
    # cfg.RSUNを使うと数字が大きすぎて無理
    RSUN = cfg.RSUN/cfg.RSUN
    
    # a_n(t)
    # TODO n=0は定義できない。
    for n in range(1, lmax):
        # 33式からAを計算
        AR += ant[n]/RSUN**(n+1)*P1n[n]
        
    for n in range(1, lmax):
        # integrate (range:0~π)
        for j in range(0, grid.jx):
            itg[n] += AR[j]*P1n[n,j]*sinth[j]* grid.dth
        # a_n(t)
        ant_2[n] = (2*n+1)*RSUN**(n+1)/(2*n*(n+1))*itg[n]
        
    # return AR, itg, ant, ant_2
    return AR, ant_2, P1n
def test_33_3(Nyquist,ant_2,P1n):
    lmax = Nyquist
    # Rでのベクトルポテンシャル（太陽内部）
    # AR = np.random.rand(grid.jx)
    AR_2 = np.zeros(grid.jx)
    # a_n(t)の項、n個あるのでルジャンドル陪関数のnに従う
    ant = ant_2[0:Nyquist]
    ant_3 = np.zeros(lmax)
    
    # 0<θ<πで定義
    sinth = np.sin(grid.th[grid.margin:grid.jxg-grid.margin])
    
    # (34)式の積分、nに従う
    itg = np.zeros(lmax)
    
    # cfg.RSUNを使うと数字が大きすぎて無理
    RSUN = cfg.RSUN/cfg.RSUN
    
    # a_n(t)
    # TODO n=0は定義できない。
    for n in range(1, lmax):
        # 33式からAを計算
        AR_2 += ant[n]/RSUN**(n+1)*P1n[n]
        
    for n in range(1, lmax):
        # integrate (range:0~π)
        for j in range(0, grid.jx):
            itg[n] += AR_2[j]*P1n[n,j]*sinth[j]* grid.dth
        # a_n(t)
        ant_3[n] = (2*n+1)*RSUN**(n+1)/(2*n*(n+1))*itg[n]
        
    # return AR_2, itg, ant, ant_3
    return AR_2

# ユークリッド距離を用いて一致度を評価
def Euclid_distance(n):
    ed = np.zeros(n)
    for i in range(2,n):
        AR, itg, ant, ant_2 = test_33_nyquist(i)
        ed2 = 0
        for j in range(2,i):
            ed2 += (ant[j] - ant_2[j])**2
        ed[i] = np.sqrt(ed2)
    return ed

# ユークリッド距離を用いて一致度を評価
def Euclid_distance_2(n):
    AR, ant_2, P1n = test_33_nyquist(n)
    ed = np.zeros(n)
    for i in range(2,n):
        AR_2 = test_33_3(i,ant_2,P1n)
        ed2 = 0
        for j in range(25,128):
            ed2 += (AR[j] - AR_2[j])**2
        ed[i] = np.sqrt(ed2)
    
    return ed

# ポアソン方程式検証コード
# 拡散方程式と形はほとんど同じ
# r tools.pyを先に実行
def poisson_test(Aph,cfg,grid):
    drr   = grid.drr
    dth   = grid.dth
    RR    = grid.RR
    RRm   = grid.RRm
    sinTH = grid.sinTH
    sinTHm = grid.sinTHm
    
    Aphrr = drr1(Aph,drr,'up')
    Aphth = dth1(Aph,dth,'up')
    
    Aph_dfrr = + drr1(RRm**2*Aphrr,drr,'dw')/RR**2
    Aph_dfth = + dth1(sinTHm*Aphth,dth,'dw')/RR**2/sinTH
    Aph_dfth = + dth1(sinTHm*Aphth,dth,'dw')/sinTH
    Aph_dfex = - Aph/RR**2/sinTH**2
    
    poisson_eq = Aph_dfrr + Aph_dfth + Aph_dfex
    
    return poisson_eq # まずmain_po.shapeが(130,130)か確認

# 検証候補：main_po[grid.ixg-1,grid.margin:grid.jxg-grid.margin]
def test_do(num):
    poisson_eq = poisson_test(Apht[:,:,num],grid)
    kk = np.zeros(130)
    for i in range(0,130):
        kk[i] = np.mean(poisson_eq[i,:])
    plt.plot(np.abs(kk))
    
def test_time_ave():
    kk = np.zeros((130,4000))
    kkk = np.zeros(130)
    for j in range(1000,5000):
        poisson_eq = poisson_test(Apht[:,:,j],grid)
        for i in range(0,130):
            kk[i,j] = np.mean(poisson_eq[i,:])
        kkk = np.mean(kk,axis=1)
    plt.plot(np.abs(kkk))

# ポテンシャル磁場境界のテスト
def test_potential(grid,legendre):
    ave, sigma = 100, 10  # 平均100, 標準偏差100
    Aph = np.random.normal(ave, sigma, (grid.ixg, grid.jxg))
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

    n_values = np.arange(1, legendre.lmax)  # n のインデックスを作成(1~127)
    coefficients = (2 * n_values + 1) / (2 * n_values * (n_values + 1)) # (127(n),)
    itg = np.sum(Aph_ex * P1n_reduced * sinth_ex * grid.dth, axis=0) # (127(n),)
    # ant の計算 (ベクトル化済み)
    ant = coefficients * itg # (127(n),)
    ant_ex = np.repeat(ant[:, np.newaxis], grid.jx, axis=1) # (127(n),128(θ))
    n_values_ex = np.repeat(n_values[:, np.newaxis], grid.jx, axis=1) # (127(n),128(θ))
    P1Sum = np.sum((n_values_ex + 1) * ant_ex / RSUN_dim * P1n_reduced.T,axis=0)
    for i in range(0, grid.margin):
        # top boundary condition
        # no electrical current
        # Bph = 0, smoothly match Aph with an exterior potential field solution
        Aph[grid.ixg-i-1,grid.margin:grid.jxg-grid.margin] = +Aph[grid.ixg-i-2,grid.margin:grid.jxg-grid.margin] \
            -grid.drr*P1Sum
        # bottom boundary condition
        # perfect conductor
        # Aph = 0, d(r*Bph)/dr = 0
        Aph[i,grid.margin:grid.jxg-grid.margin] = - Aph[2*grid.margin-i-1,grid.margin:grid.jxg-grid.margin]
    return Aph

def test_potential_33(grid,legendre):

    Aph = Apht[:,:,4500]
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

    n_values = np.arange(1, legendre.lmax)  # n のインデックスを作成(1~127)
    coefficients = (2 * n_values + 1) / (2 * n_values * (n_values + 1)) # (127(n),)
    itg = np.sum(Aph_ex * P1n_reduced * sinth_ex * grid.dth, axis=0) # (127(n),)
    # ant の計算 (ベクトル化済み)
    ant = coefficients * itg # (127(n),)
    ant_ex = np.repeat(ant[:, np.newaxis], grid.jx, axis=1) # (127(n),128(θ))
    n_values_ex = np.repeat(n_values[:, np.newaxis], grid.jx, axis=1) # (127(n),128(θ))
    P1Sum = np.sum((n_values_ex + 1) * ant_ex / RSUN_dim * P1n_reduced.T,axis=0)
    for i in range(0, grid.margin):
        # top boundary condition
        # no electrical current
        # Bph = 0, smoothly match Aph with an exterior potential field solution
        Aph[grid.ixg-i-1,grid.margin:grid.jxg-grid.margin] = +Aph[grid.ixg-i-2,grid.margin:grid.jxg-grid.margin] \
            -grid.drr*P1Sum
        # bottom boundary condition
        # perfect conductor
        # Aph = 0, d(r*Bph)/dr = 0
        Aph[i,grid.margin:grid.jxg-grid.margin] = - Aph[2*grid.margin-i-1,grid.margin:grid.jxg-grid.margin]
    return Aph

def test_potential_2(grid,legendre):
    ave, sigma = 0, 1e13  # 平均100, 標準偏差100
    Aph = np.random.normal(ave, sigma, (grid.ixg, grid.jxg))
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
    for i in range(0, grid.margin):
        # top boundary condition
        # no electrical current
        # Bph = 0, smoothly match Aph with an exterior potential field solution
        Aph[grid.ixg-i-1,grid.margin:grid.jxg-grid.margin] = +Aph[grid.ixg-i-2,grid.margin:grid.jxg-grid.margin] \
            -grid.drr*P1Sum
        
        Aph[grid.ixg-i-1,grid.margin:grid.jxg-grid.margin] = np.sum(ant_ex * P1n_reduced,axis = 0)
        # bottom boundary condition
        # perfect conductor
        # Aph = 0, d(r*Bph)/dr = 0
        Aph[i,grid.margin:grid.jxg-grid.margin] = - Aph[2*grid.margin-i-1,grid.margin:grid.jxg-grid.margin]
    return Aph,P1Sum

# ave, sigma = 0, 1e6  # 平均100, 標準偏差100
# Aph = np.random.normal(ave, sigma, (grid.ixg, grid.jxg))

def survey_simulation(cfg=None, parameter_file=None):
    so0_list = [5.0, 10.0, 30.0, 50.0, 70.0]
    uu0_list = [250, 500, 750, 1000, 1500 ,2000]   
    if cfg is None:
        if parameter_file is None:
            cfg = S2MFD.Cfg()
        else:
            cfg = S2MFD.Cfg(parameter_file)
    ii = 0
    jj = 0
    print(cfg)
    for ii in range(0, 5):
        cfg.so0 = so0_list[ii]
        for jj in range(0, 5):
            cfg.uu0 = uu0_list[jj]
            cfg.datadir = f"data_{ii}{jj}/"
            sim = S2MFD.Simulation(cfg)
            sim.initialize_simulation()
            sim.cfl_condition()
            sim.initial_condition()
            sim.main_loop()

def analysis_sample():
    for ii in range(0, 5):
        for jj in range(0, 4):
            datadir = f"data_{ii}{jj}/"
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

            n0 = 2000
            tau_diff = data.cfg.RSUN**2/data.cfg.ett
            timet = np.zeros(n1-n0)
            Brrt = np.zeros((grid.ixg,grid.jxg,n1-n0))
            Btht = np.zeros((grid.ixg,grid.jxg,n1-n0))
            Bpht = np.zeros((grid.ixg,grid.jxg,n1-n0))
            Apht = np.zeros((grid.ixg,grid.jxg,n1-n0))

            for n  in range(n0,n1):
                # print(n)
                data.data_load(n)
                Brr, Bth = S2MFD.physics.poloidal_mag(data.Aph, grid.RR, grid.sinTH, grid.drr, grid.dth)
                d = np.load(file=datadir+'data.'+str(n).zfill(6)+'.npz')
                timet[n-n0] = d['time']
                Brrt[:,:,n-n0] = Brr
                Btht[:,:,n-n0] = Bth
                Bpht[:,:,n-n0] = d['Bph']
                Apht[:,:,n-n0] = d['Aph']

            Bpht0 = Bpht[1+np.argmin(abs(grid.rr-0.7*cfg.RSUN)),np.argmin(abs(grid.th-30/180*np.pi)),:]
            Brrt0 = Brrt[-2,np.argmin(abs(grid.th-60/180*np.pi)),:]

            Bpht0_sign = np.sign(Bpht0)
            Bpht0_sign_diff = np.diff(Bpht0_sign)

            Brrt0_sign = np.sign(Brrt0)
            Brrt0_sign_diff = np.diff(Brrt0_sign)
            ne_r = np.where(Brrt0_sign_diff == +2)[0][-1]


            ns = np.where(Bpht0_sign_diff == +2)[0][-2]
            ne = np.where(Bpht0_sign_diff == +2)[0][-1]

            # plt.clf()
            # plt.close('all')
            # fig = plt.figure('J08_test',figsize=(6,10))
            # ax1 = fig.add_subplot(2,1,1)
            # ax2 = fig.add_subplot(2,1,2)

            timeu = (timet[ns:ne]-timet[ns])/tau_diff
            timeur = (timet[ns:ne_r]-timet[ns])/tau_diff
            Bpht0u = Bpht0[ns:ne]
            Brrt0u = Brrt0[ns:ne]
            # nw = ne - ns
            # nm = np.argmax(Bpht0u)

            # ax1.plot(timeu,Bpht0u)
            # ax2.plot(timeu,Brrt0u)

            # ax1.set_ylabel(r'$B_\phi$: $r=0.7R_\odot$, $\theta=30^\circ$')
            # ax2.set_ylabel(r'$B_r$: $r=R_\odot$, $\theta=60^\circ$')

            # ax2.set_xlabel(r't/$\tau_\mathrm{diff}$')

            # fig.tight_layout()
            # plt.savefig("output.png")
            print(ii,jj)
            # print('Cycle time = ',timeu[-1])
            # print('Cycle time = ',timeur[-1])
            print('Period(year)(surface) = ',timeu[-1]*tau_diff/86400/365,'year')
            print('Max(Bph(0.7R,30)) =',np.max(Bpht0u))
            print('Max(Brr(1.0R,60)) =',np.max(Brrt0u))
            # 以下3つはnp.zeros(())で初期化しておく
            Period[ii,jj] = timeu[-1]*tau_diff/86400/365
            BphAmp[ii,jj] = np.max(Bpht0u)
            BrrAmp[ii,jj] = np.max(Brrt0u) 
            
            
# ========================================================================================== #
# 並列化解析_worker
import multiprocessing
def analysis_worker(ii, jj, Period_shared, BphAmp_shared, BrrAmp_shared):
    so0_list = [5.0, 10.0, 30.0, 50.0, 70.0]
    uu0_list = [250, 500, 750, 1000, 1500]
    
    datadir = f"data_{ii}{jj}/"
    
    if not os.path.isdir(datadir):
        print(f"Skipping {datadir}: Directory does not exist")
        return
    
    data = S2MFD.Data.initial_load(datadir)
    cfg = data.cfg
    grid = data.grid
    setup = data.setup

    fig = plt.figure('dynamo', figsize=(10,10)) 

    n1 = 0
    files = os.listdir(datadir)
    for file in files:
        filel = file.split('.')
        if filel[0] == 'data':
            n1 = max(n1, int(filel[1]))

    n0 = 2000
    tau_diff = data.cfg.RSUN**2 / data.cfg.ett
    timet = np.zeros(n1 - n0)
    Brrt = np.zeros((grid.ixg, grid.jxg, n1 - n0))
    Btht = np.zeros((grid.ixg, grid.jxg, n1 - n0))
    Bpht = np.zeros((grid.ixg, grid.jxg, n1 - n0))
    Apht = np.zeros((grid.ixg, grid.jxg, n1 - n0))

    for n in range(n0, n1):
        data.data_load(n)
        Brr, Bth = S2MFD.physics.poloidal_mag(data.Aph, grid.RR, grid.sinTH, grid.drr, grid.dth)
        d = np.load(file=datadir + 'data.' + str(n).zfill(6) + '.npz')
        timet[n - n0] = d['time']
        Brrt[:, :, n - n0] = Brr
        Btht[:, :, n - n0] = Bth
        Bpht[:, :, n - n0] = d['Bph']
        Apht[:, :, n - n0] = d['Aph']

    Bpht0 = Bpht[1 + np.argmin(abs(grid.rr - 0.7 * cfg.RSUN)), np.argmin(abs(grid.th - 30 / 180 * np.pi)), :]
    Brrt0 = Brrt[-2, np.argmin(abs(grid.th - 60 / 180 * np.pi)), :]

    Bpht0_sign = np.sign(Bpht0)
    Bpht0_sign_diff = np.diff(Bpht0_sign)
    Brrt0_sign = np.sign(Brrt0)
    Brrt0_sign_diff = np.diff(Brrt0_sign)

    ne_r = np.where(Brrt0_sign_diff == +2)[0][-1]
    ns = np.where(Bpht0_sign_diff == +2)[0][-2]
    ne = np.where(Bpht0_sign_diff == +2)[0][-1]

    timeu = (timet[ns:ne] - timet[ns]) / tau_diff
    timeur = (timet[ns:ne_r] - timet[ns]) / tau_diff
    Bpht0u = Bpht0[ns:ne]
    Brrt0u = Brrt0[ns:ne]

    print(f"Processed: ii={ii}, jj={jj}")
    print('Period(year)(surface) = ', timeu[-1] * tau_diff / 86400 / 365, 'year')
    print('Max(Bph(0.7R,30)) =', np.max(Bpht0u))
    print('Max(Brr(1.0R,60)) =', np.max(Brrt0u))

    # 共有メモリの配列に書き込み
    Period_shared[ii][jj] = timeu[-1] * tau_diff / 86400 / 365
    BphAmp_shared[ii][jj] = np.max(Bpht0u)
    BrrAmp_shared[ii][jj] = np.max(Brrt0u)
# ========================================================================================== #


# ========================================================================================== #
# 並列解析メイン関数
def main_parallel():
    so0_list = [5.0, 10.0, 30.0, 50.0, 70.0]
    uu0_list = [250, 500, 750, 1000]
    num_processes = min(os.cpu_count(), len(so0_list)*len(uu0_list))

    Period = np.zeros((len(so0_list), len(uu0_list)))
    BphAmp = np.zeros((len(so0_list), len(uu0_list)))
    BrrAmp = np.zeros((len(so0_list), len(uu0_list)))

    with multiprocessing.Manager() as manager:
        Period_shared = manager.list([manager.list([0] * len(uu0_list)) for _ in range(len(so0_list))])
        BphAmp_shared = manager.list([manager.list([0] * len(uu0_list)) for _ in range(len(so0_list))])
        BrrAmp_shared = manager.list([manager.list([0] * len(uu0_list)) for _ in range(len(so0_list))])

        task_list = [(ii, jj, Period_shared, BphAmp_shared, BrrAmp_shared) for ii in range(len(so0_list)) for jj in range(len(uu0_list))]

        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.starmap(analysis_worker, task_list)

        for ii in range(len(so0_list)):
            for jj in range(len(uu0_list)):
                Period[ii, jj] = Period_shared[ii][jj]
                BphAmp[ii, jj] = BphAmp_shared[ii][jj]
                BrrAmp[ii, jj] = BrrAmp_shared[ii][jj]

    print("Period:\n", Period)
    print("BphAmp:\n", BphAmp)
    print("BrrAmp:\n", BrrAmp)
    
    return Period, BphAmp, BrrAmp
# ========================================================================================== #