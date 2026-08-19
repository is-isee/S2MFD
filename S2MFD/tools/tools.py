import numpy as np
from numba import njit

@njit
def drr1(qq,drr,dir):
    '''
    To calculate 1st order accuracy derivative in r direction

    Parameters
    ----------
    qq : numpy.ndarray, float
        Quantity to be differentiated (2D)
    drr : float
        Grid spacing in r direction
    dir: string
        'up' or 'dw'

        'up': dqq[i] = (qq[i] - qq[i-1])/drr  (後退差分、i=1..N-1 に格納)
        'dw': dqq[i] = (qq[i+1] - qq[i])/drr  (前進差分、i=0..N-2 に格納)

        セル境界 i-1/2 の勾配を 'up' で作り、'dw' で発散を取ると
        保存形の2階差分になる (diffusion() での用法)。
    Returns
    -------
    dqq: numpy.ndarray, float
        Differentiated quantity (2D)
    '''

    if dir == 'up':
        i0, i1 = 1, qq.shape[0]
    elif dir == 'dw':
        i0, i1 = 0, qq.shape[0]-1
    else:
        raise ValueError('dir must be up or dw')

    jnum = qq.shape[1]
    dqq = np.zeros_like(qq)
    for i in range(i1-i0):
        for j in range(jnum):
            dqq[i+i0,j] = (qq[i+1,j] - qq[i,j])/drr

    return dqq

@njit
def drr2(qq,drr):
    '''
    To calculate 2nd order accuracy derivative in r direction
   
    Parameters
    ----------
    qq: numpy.ndarray, float
        Quantity to be differentiated (2D)
    drr: float
        Grid spacing in r direction
    Returns
    -------
        dqq: numpy.ndarray, float
            Differentiated quantity (2D)
    '''

    inum = qq.shape[0]
    jnum = qq.shape[1]
    dqq = np.zeros_like(qq)
    for i in range(inum-2):
        for j in range(jnum):
            dqq[i+1,j] = (qq[i+2,j] - qq[i,j])/drr*0.5
   
    return dqq

@njit
def dth1(qq,dth,dir):
    '''
    To calculate 1st order accuracy derivative in theta direction
   
    Parameters
    ----------
        qq: numpy.ndarray, float
            Quantity to be differentiated (2D)
        dth: float
            Grid spacing in theta direction
        dir: string
            'up' or 'dw'

            'up': dqq[j] = (qq[j] - qq[j-1])/dth  (後退差分、j=1..N-1 に格納)
            'dw': dqq[j] = (qq[j+1] - qq[j])/dth  (前進差分、j=0..N-2 に格納)

    Returns
    -------
        dqq: numpy.ndarray
            Differentiated quantity (2D)
    '''

    if dir == 'up':
        j0, j1= 1,qq.shape[1]
    elif dir == 'dw':
        j0, j1 = 0, qq.shape[1]-1
    else:
        raise ValueError('dir must be up or dw')

    inum = qq.shape[0]
    dqq = np.zeros_like(qq)
    for i in range(inum):
        for j in range(j1-j0):
            dqq[i,j+j0] = (qq[i,j+1] - qq[i,j])/dth
        
    return dqq

@njit
def dth2(qq,dth):
    '''
    To calculate 2nd order accuracy derivative in theta direction
   
    Parameters
    ----------
        qq: numpy.ndarray
            Quantity to be differentiated (2D)
        dth: float
            Grid spacing in theta direction

    Returns
    -------
        dqq: differentiated quantity (2D)
    '''

    inum = qq.shape[0]
    jnum = qq.shape[1]
    dqq = np.zeros_like(qq)
    for i in range(inum):
        for j in range(jnum-2):
            dqq[i,j+1] = (qq[i,j+2] - qq[i,j])/dth*0.5

    return dqq

def sunspot_proxy(Bph, grid, cfg, gamma=5.8653520852, r_frac=0.7, theta_deg=75.0):
    '''
    黒点数プロキシ SN = gamma * Bph(r = r_frac*RSUN, theta = theta_deg)^2

    Shimizu & Hotta (2026) 式 (3.5) 相当。gamma は平均的な太陽極大期の
    黒点数 166.5 に合うよう較正された値。

    Parameters
    ----------
    Bph : numpy.ndarray
        Longitudinal magnetic field (2D)
    grid : S2MFD.Grid
        Grid object
    cfg : S2MFD.Cfg
        Configuration object (RSUN を参照)
    gamma : float
        較正係数
    r_frac : float
        参照半径 (RSUN 単位)
    theta_deg : float
        参照余緯度 (度)

    Returns
    -------
    float
        黒点数プロキシ
    '''
    base = 1 + np.argmin(abs(grid.rr - r_frac*cfg.RSUN))
    loca = np.argmin(abs(grid.th - theta_deg/180*np.pi))
    return gamma * Bph[base, loca]**2
