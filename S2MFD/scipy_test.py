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
    ale = np.zeros(50)
    for m in range(0,50): 
        for i in range(grid.margin, grid.jxg - grid.margin):
            ale[m] += legendre.P1n[n,i]*legendre.P1n[m,i]*grid.sinTH[0,i]*grid.dth
    return ale
# Dikpati+1994(33)式の検証
def test_33():
    # Rでのベクトルポテンシャル
    AR = np.linspace(0,grid.jx,128)
    ARe = np.zeros(grid.jx)
    
    # a_n(t)の項、n個あるのでルジャンドル陪関数のnに従う
    ant = np.zeros(legendre.termnum)
    
    # 0<θ<πで定義
    sinth = np.sin(grid.th[grid.margin:grid.jxg-grid.margin])
    
    # (34)式の積分、nに従う
    itg = np.zeros(legendre.termnum)
    
    # cfg.RSUNを使うと数字が大きすぎて無理
    RSUN = cfg.RSUN/cfg.RSUN
    
    # a_n(t)
    # TODO n=0は定義できない。
    for n in range(1, legendre.termnum):
        # integrate (range:0~π)
        for j in range(0, grid.jx):
            itg[n] += AR[j]*legendre.P1n[n,j]*sinth[j]* grid.dth
        # a_n(t)
        ant[n] = (2*n+1)*RSUN**(n+1)/(2*n*(n+1))*itg[n]
        # 1~nまで足しあげる（Σ）
        ARe += ant[n]/RSUN**(n+1)*legendre.P1n[n]
    return AR, ARe, itg, ant

# Dikpati+1994(33)式の検証
def test_33_an():
    # Rでのベクトルポテンシャル（太陽内部）
    # AR = np.random.rand(grid.jx)
    AR = np.zeros(grid.jx)
    # a_n(t)の項、n個あるのでルジャンドル陪関数のnに従う
    ant = np.random.randn(legendre.termnum)
    ant_2 = np.zeros(legendre.termnum)
    
    # 0<θ<πで定義
    sinth = np.sin(grid.th[grid.margin:grid.jxg-grid.margin])
    
    # (34)式の積分、nに従う
    itg = np.zeros(legendre.termnum)
    
    # cfg.RSUNを使うと数字が大きすぎて無理
    RSUN = cfg.RSUN/cfg.RSUN
    
    # a_n(t)
    # TODO n=0は定義できない。
    for n in range(1, legendre.termnum):
        # 33式からAを計算
        AR += ant[n]/RSUN**(n+1)*legendre.P1n[n]
        
    for n in range(1, legendre.termnum):
        # integrate (range:0~π)
        for j in range(0, grid.jx):
            itg[n] += AR[j]*legendre.P1n[n,j]*sinth[j]* grid.dth
        # a_n(t)
        ant_2[n] = (2*n+1)*RSUN**(n+1)/(2*n*(n+1))*itg[n]
        
    return AR, itg, ant, ant_2

# ナイキスト波数の検討
# Nyquistで第何項まで書くか指定
def test_33_nyquist(Nyquist):
    # 0<θ<πで定義
    costh = np.cos(grid.th[grid.margin:grid.jxg-grid.margin])
    termnum = Nyquist
    P1n = np.zeros((termnum,grid.jx))
    # n = 1
    P1n[1,:] = -(1-costh**2)**0.5
    # n = 2, 3, ...
    for i in range(2, termnum):
        # recurrence relation
        P1n[i,:] = ((2*i-1)/(i-1))*costh*P1n[i-1,:]-(i/(i-1))*P1n[i-2,:]
    # Rでのベクトルポテンシャル（太陽内部）
    # AR = np.random.rand(grid.jx)
    AR = np.zeros(grid.jx)
    # a_n(t)の項、n個あるのでルジャンドル陪関数のnに従う
    ant = np.random.rand(termnum)
    ant_2 = np.zeros(termnum)
    
    # 0<θ<πで定義
    sinth = np.sin(grid.th[grid.margin:grid.jxg-grid.margin])
    
    # (34)式の積分、nに従う
    itg = np.zeros(termnum)
    
    # cfg.RSUNを使うと数字が大きすぎて無理
    RSUN = cfg.RSUN/cfg.RSUN
    
    # a_n(t)
    # TODO n=0は定義できない。
    for n in range(1, termnum):
        # 33式からAを計算
        AR += ant[n]/RSUN**(n+1)*P1n[n]
        
    for n in range(1, termnum):
        # integrate (range:0~π)
        for j in range(0, grid.jx):
            itg[n] += AR[j]*P1n[n,j]*sinth[j]* grid.dth
        # a_n(t)
        ant_2[n] = (2*n+1)*RSUN**(n+1)/(2*n*(n+1))*itg[n]
        
    # return AR, itg, ant, ant_2
    return AR, ant_2, P1n
def test_33_3(Nyquist,ant_2,P1n):
    termnum = Nyquist
    # Rでのベクトルポテンシャル（太陽内部）
    # AR = np.random.rand(grid.jx)
    AR_2 = np.zeros(grid.jx)
    # a_n(t)の項、n個あるのでルジャンドル陪関数のnに従う
    ant = ant_2[0:Nyquist]
    ant_3 = np.zeros(termnum)
    
    # 0<θ<πで定義
    sinth = np.sin(grid.th[grid.margin:grid.jxg-grid.margin])
    
    # (34)式の積分、nに従う
    itg = np.zeros(termnum)
    
    # cfg.RSUNを使うと数字が大きすぎて無理
    RSUN = cfg.RSUN/cfg.RSUN
    
    # a_n(t)
    # TODO n=0は定義できない。
    for n in range(1, termnum):
        # 33式からAを計算
        AR_2 += ant[n]/RSUN**(n+1)*P1n[n]
        
    for n in range(1, termnum):
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