from dataclasses import dataclass, field
import numpy as np

from S2MFD.npz_io import NpzIO


@dataclass
class Grid(NpzIO):
   """
   Class for managing the grid data.

   Attributes
   ----------
   ix : int
      Number of grid points in the radial direction.
   jx : int
      Number of grid points in the colatitudinal direction.
   margin : int
      Number of margin points on each side of the grid.
   ixg : int
      ix + 2*margin.
   jxg : int
      jx + 2*margin.
   rrmin : float
      Minimum value for the radial direction.
   rrmax : float
      Maximum value for the radial direction.
   thmin : float
      Minimum value for the colatitudinal direction.
   thmax : float
      Maximum value for the colatitudial direction.
   drr : float
      Radial grid spacing.
   dth : float
      Colatitudinal grid spacing.
   rr : numpy.ndarray
      Array of radial grid points.
   th : numpy.ndarray
      Array of colatitudinal grid points.
   RR : numpy.ndarray: 
      Radial coordinate np.meshgrid array.
   TH : numpy.ndarray
      Colatitudinal coordinate np.meshgrid array.
   RRm : numpy.ndarray
      Radial coordinate meshgrid array for centered points.
   THm : numpy.ndarray
      Colatitudinal coordinate meshgrid array for centered points.
   sinTH : numpy.ndarray
      numpy.sin(TH)
   cosTH : numpy.ndarray
      numpy.cos(TH)
   sinTHm : numpy.ndarray
      numpy.sin(THm) (face-centered)
   X : numpy.ndarray
      Cartesian x-coordinates based on radial and colatitudinal grids.
   Y : numpy.ndarray
      Cartesian y-coordinates based on radial and colatitudinal grids.
   """
   ix: int
   jx: int
   ixg: int = field(init=False)
   jxg: int = field(init=False)
   margin: int
   rrmin: float
   rrmax: float
   thmin: float
   thmax: float
   drr: float = field(init=False)
   dth: float = field(init=False)
   rr: np.ndarray = field(init=False)
   th: np.ndarray = field(init=False)
   
   RR: np.ndarray = field(init=False)
   TH: np.ndarray = field(init=False)
   RRm: np.ndarray = field(init=False) 
   THm: np.ndarray = field(init=False)
   sinTH: np.ndarray = field(init=False)
   cosTH: np.ndarray = field(init=False)
   sinTHm: np.ndarray = field(init=False)
   X: np.ndarray = field(init=False)
   Y: np.ndarray = field(init=False)
   
   def __post_init__(self):
      # dr,dθの設定
      self.ixg = self.ix + 2*self.margin
      self.jxg = self.jx + 2*self.margin
      self.drr = (self.rrmax - self.rrmin)/self.ix
      self.dth = (self.thmax - self.thmin)/self.jx

      #座標rrの設定 (セル中心、ゴーストセル込み)
      rr0 = self.rrmin + self.drr*(0.5 - self.margin)
      self.rr = rr0 + self.drr*np.arange(self.ixg)

      #座標thの設定
      th0 = self.thmin + self.dth*(0.5 - self.margin)
      self.th = th0 + self.dth*np.arange(self.jxg)
         
      self.RR ,self.TH  = np.meshgrid(self.rr, self.th,indexing='ij')
      self.RRm = np.zeros_like(self.RR)
      self.THm = np.zeros_like(self.TH)
      
      self.RRm[1:self.ixg,:] = 0.5*(self.RR[1:self.ixg,:] + self.RR[0:self.ixg-1,:])
      self.THm[:,1:self.jxg] = 0.5*(self.TH[:,1:self.jxg] + self.TH[:,0:self.jxg-1])
      
      self.sinTH = np.sin(self.TH)
      self.cosTH = np.cos(self.TH)
      self.sinTHm = np.sin(self.THm)
      
      self.X, self.Y = self.RR * np.cos(self.TH), self.RR * np.sin(self.TH)