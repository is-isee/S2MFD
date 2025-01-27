from dataclasses import dataclass, field
from S2MFD.tools import drr2, dth2
import S2MFD.cfg as Cfg
import numpy as np
import pickle
from scipy.special import erf

@dataclass
class Legendre:
    """
    Class for managing the legendre.

    Attributes
    ----------
    cosTH : numpy.ndarray
        numpy.cos(TH)
    P1n : numpy.ndarray
        associated legendre polynomials 
    """
    def __init__(self,grid):
        # 0<θ<πで定義
        self.costh = np.cos(grid.th[grid.margin:grid.jxg-grid.margin])
        self.lmax = grid.jx
        self.P1n = np.zeros((self.lmax,grid.jx))
        # n = 1
        self.P1n[1,:] = -(1-self.costh**2)**0.5
        # n = 2, 3, ...
        for i in range(2, self.lmax):
            # recurrence relation
            self.P1n[i,:] = ((2*i-1)/(i-1))*self.costh*self.P1n[i-1,:]-(i/(i-1))*self.P1n[i-2,:]