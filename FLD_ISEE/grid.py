from dataclasses import dataclass, field
import numpy as np
import pickle

@dataclass
class grid_c:
   """
    grid_c class represents a grid configuration for a simulation, defining the grid points in
    both radial and angular coordinates, and computing associated trigonometric and Cartesian
    transformations.

    Attributes:
        ix (int): Number of grid points in the radial direction.
        jx (int): Number of grid points in the colatitudinal direction.
        ixg (int): ix + 2*margin.
        jxg (int): jx + 2*margin.
        margin (int): Number of margin points on each side of the grid.
        rrmin (float): Minimum value for the radial direction.
        rrmax (float): Maximum value for the radial direction.
        thmin (float): Minimum value for the colatitudinal direction.
        thmax (float): Maximum value for the colatitudial direction.
        drr (float): Radial grid spacing.
        dth (float): Angular grid spacing.
        rr (np.ndarray): Array of radial grid points.
        th (np.ndarray): Array of colatitudinal grid points.
        RR (np.ndarray): Radial coordinate np.meshgrid array.
        TH (np.ndarray): Colatitudinal coordinate np.meshgrid array.
        RRm (np.ndarray): Radial coordinate meshgrid array for centered points.
        THm (np.ndarray): Colatitudinal coordinate meshgrid array for centered points.
        sinTH (np.ndarray): np.sin(TH)
        cosTH (np.ndarray): np.cos(TH)
        X (np.ndarray): Cartesian x-coordinates based on radial and colatitudinal grids.
        Y (np.ndarray): Cartesian y-coordinates based on radial and colatitudinal grids.

    Methods:
        __post_init__(): Initializes the grid coordinates and related arrays after object creation.
        save(filename): Saves the grid_c instance to a file.
        load(filename): Loads a grid_c instance from a file.
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
   X: np.ndarray = field(init=False)
   Y: np.ndarray = field(init=False)
   
   def __post_init__(self):
      # dr,dθの設定
      self.ixg = self.ix + 2*self.margin
      self.jxg = self.jx + 2*self.margin
      self.drr = (self.rrmax - self.rrmin)/self.ix
      self.dth = (self.thmax - self.thmin)/self.jx

      #座標rrの設定
      self.rr = np.zeros(self.ixg)
      self.rr[0] = self.rrmin + self.drr*(0.5 - self.margin)

      for i in range(1, self.ixg):
         self.rr[i] = self.rr[i - 1] + self.drr
   
      #座標thの設定    
      self.th    = np.zeros(self.jxg)
      self.th[0] = self.thmin + self.dth*(0.5 - self.margin)

      for j in range(1,self.jxg):
         self.th[j] = self.th[j - 1] + self.dth
         
      self.RR ,self.TH  = np.meshgrid(self.rr, self.th,indexing='ij')
      self.RRm = np.zeros_like(self.RR)
      self.THm = np.zeros_like(self.TH)
      
      self.RRm[1:self.ixg,:] = 0.5*(self.RR[1:self.ixg,:] + self.RR[0:self.ixg-1,:])
      self.THm[:,1:self.jxg] = 0.5*(self.TH[:,1:self.jxg] + self.TH[:,0:self.jxg-1])
      
      self.sinTH = np.sin(self.TH)
      self.cosTH = np.cos(self.TH)
      self.sinTHm = np.sin(self.THm)
      
      self.X, self.Y = self.RR * np.cos(self.TH), self.RR * np.sin(self.TH)
   
   def save(self, filename):
      with open(filename, 'wb') as f:
         pickle.dump(self, f)
      print('grid_c instance saved to', filename)
   
   @classmethod
   def load(cls, filename):
      with open(filename, 'rb') as f:
         return pickle.load(f)