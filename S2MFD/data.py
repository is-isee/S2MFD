class S2MFD_data:
    """
    Class to handle S2MFD data including configuration, grid, and setup.

    Attributes:
    cfg (object): S2MFD.config_c object.
    grid (object): S2MFD.grid_c object.
    setup (object): S2MFD.setup_c object.
    Bph (ndarray): Longitudinal magnetic field.
    Aph (ndarray): Longitudinal vector potential.
    time (float): simulation time
    dt (float): Time spacing
    n (int): Time step
    nd (int): Data output step 
    """
    def __init__(self, cfg, grid, setup):
        """
        Initialize the S2MFD_data object.

        Parameters:
        cfg (object): S2MFD.config_c object.
        grid (object): S2MFD.grid_c object.
        setup (object): S2MFD.setup_c object.
        """
        self.cfg = cfg
        self.grid = grid
        self.setup = setup

        self.Bph = None
        self.Aph = None
        self.time = None
        self.dt = None
        self.n = None
        self.nd = None      
    
    @classmethod
    def initial_load(cls,datadir):
        """ 
        Load initial configuration, grid, and setup from the specified directory.

        Parameters:
        datadir (str): Directory containing the configuration, grid, and setup files.

        Returns:
        S2MFD_data: An instance of the S2MFD_data class.
        """        
        cfg = S2MFD.config_c.load(datadir+'config.json')
        grid = S2MFD.grid_c.load(cfg.gridfile)
        setup = S2MFD.setup_c.load(cfg.setupfile)
        
        return cls(cfg,grid,setup)
    
    def get_data_file_path(self, nd):
        """
        Generate the file path for a specific step.

        Parameters:
        nd (int): Data output step 

        Returns:
        str: File path for the specified data step.
        """        
        return self.cfg.datadir+'data.'+str(nd).zfill(6)+'.npz'
    
    def data_load(self,nd):
        """
        Load data from a file for a specific step.

        Parameters:
        nd (int): Data output step 
        """        
        filename = self.get_data_file_path(nd)
        d = np.load(file=filename)
        self.Bph = d['Bph']
        self.Aph = d['Aph']
        self.time = d['time']
        self.n = d['n']
        self.nd = nd