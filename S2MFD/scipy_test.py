import scipy.special
def scipy_le():
    le = np.zeros((len(grid.cosTH), 100))
    for i in range(0,100):
        le[:,i] = scipy.special.lpmv(1,i,grid.cosTH[0,:])
    # plt.plot(grid.cosTH[0,:],le[:,20])
    return le

# 直交性の検証

def orthogonality(n):
    ale = np.zeros(50)
    for m in range(0,50): 
        for i in range(grid.margin, grid.jxg - grid.margin):
            ale[m] += legendre.Pln[i,n]*legendre.Pln[i,m]*grid.sinTH[0,i]*grid.dth
    return ale