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
    Pln : numpy.ndarray
        associated legendre polynomials 
    """
    def __init__(self,grid):
        self.cosTH = grid.cosTH
        self.termnum = int(grid.ixg*0.5)
        self.Pln = np.zeros((len(self.cosTH), self.termnum))
        # n = 1
        self.Pln[:,1] = -(1-self.cosTH[0,:]**2)**0.5
        # n = 2, 3, ...
        for i in range(2, self.termnum):
            # recurrence relation
            self.Pln[:,i] = ((2*i-1)/(i-1))*self.cosTH[0,:]*self.Pln[:,i-1]-(i/(i-1))*self.Pln[:,i-2]