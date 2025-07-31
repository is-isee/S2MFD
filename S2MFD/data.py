import S2MFD
import numpy as np

class Data:
    """
    Class for managing the whole simulation data.

    Attributes
    --------------
    cfg : S2MFD.Cfg
      Configuration object.
    grid : S2MFD.Grid
      Grid object.
    setup : S2MFD.Setup
      Setup object.
    Bph : numpy.ndarray
      Longitudinal magnetic field.
    Aph : numpy.ndarray
      Longitudinal vector potential.
    time : float
      Simulation time.
    dt : float
      Time spacing.
    n : int
      Time step.
    nd : int
      Data output step.
    """
    def __init__(self, cfg, grid, setup, legendre):
        """
        Initialize the S2MFD_data object.
        
        Parameters
        ----------
        cfg : S2MFD.Cfg
            Configuration object.
        grid : S2MFD.Grid
            Grid object.
        setup : S2MFD.Setup
            Setup object.
            
        Returns
        -------
        None     
        """
        self.cfg = cfg
        self.grid = grid
        self.setup = setup
        self.legendre = legendre

        self.Aph = None
        self.time = None
        self.dt = None
        self.n = None
        self.nd = None      
    
    @classmethod
    def initial_load(cls,datadir):
        """ 
        Load initial configuration, grid, and setup from the specified directory.

        Parameters
        ----------
        datadir : str
            Directory containing the configuration, grid, and setup files.

        Returns
        -------
        S2MFD.S2MFD_data
            An instance of the S2MFD_data class.
        """        
        cfg = S2MFD.Cfg.load(datadir+'config.json')
        cfg.datadir = datadir
        grid = S2MFD.Grid.load(datadir+cfg.gridfile)
        setup = S2MFD.Setup.load(datadir+cfg.setupfile)
        legendre = S2MFD.Legendre.load(datadir+cfg.legendrefile)
        
        return cls(cfg,grid,setup,legendre)
    
    def get_data_file_path(self, nd):
        """
        Generate the file path for a specific step.

        Parameters
        ----------
        nd : int
            Data output step 

        Returns
        -------
        str
            File path for the specified data step.
        """        
        return self.cfg.datadir+'data.'+str(nd).zfill(6)+'.npz'
    
    def data_load(self,nd):
        """
        Load data for `Bph`, `Aph`, `time`, `n` and `nd`, from a file for a specific step.

        Parameters
        ----------
        nd : int
            Data output step 
        """        
        filename = self.get_data_file_path(nd)
        d = np.load(file=filename, allow_pickle=True)
        self.Bph = d['Bph']
        self.Aph = d['Aph']
        self.time = d['time']
        self.n = d['n']
        self.nd = nd