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
    # Rでのベクトルポテンシャル（太陽内部）
    AR = np.random.rand(grid.jx)
    # Rでのベクトルポテンシャル（太陽外部）
    ARe = np.zeros(grid.jx)
    
    # a_n(t)の項、n個あるのでルジャンドル陪関数のnに従う
    ale = np.zeros(legendre.termnum)
    
    # 0<θ<πで定義
    sinth = np.sin(grid.th[grid.margin:grid.jxg-grid.margin])
    
    # (34)式の積分、nに従う
    itg = np.zeros(legendre.termnum)
    
    # TODO cfg.RSUNを使うと数字が大きすぎて無理
    # RSUN = cfg.RSUN
    RSUN = 2
    
    # a_n(t)
    # TODO n=0は定義できない。
    for i in range(1, legendre.termnum):
        # integrate (range:0~π)
        for j in range(0, grid.jx):
            itg[i] += AR[j]*legendre.P1n[i,j]*sinth[j]* grid.dth
        # a_n(t)
        ale[i] = (2*i+1)*RSUN**(i+1)/(2*i*(i+1))*itg[i]
        # 1~nまで足しあげる（Σ）
        ARe += ale[i]/RSUN**(i+1)*legendre.P1n[i]
    return AR, ARe, itg, ale

