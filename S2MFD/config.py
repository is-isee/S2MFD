import numpy as np
import json

class config_c:
    def __init__(self):
        # fixed parameters (modifications are not recommended)
        self.margin = 1
        self.RSUN = 6.96e10

        # time parameter
        d2s = 86400 # day to second
        self.tend = 30000*d2s
        self.dtout = 100*d2s

        # geometry parameters
        self.ix = 128 # number of grid points in r-direction
        self.jx = 128 # number of grid points in theta-direction
        self.rrmin = 0.65*self.RSUN
        self.rrmax = self.RSUN
        self.thmin = 0
        self.thmax = np.pi

        # Setup parameters
        ## geometry parameters
        self.rrc = 0.7*self.RSUN  # base of the convection zone
        self.d   = 0.02*self.RSUN # width of the tachocline

        ## Differential rotation
        self.ome = 456.e-9*2*np.pi # rotation rate at equator
        self.omc = 0.92*self.ome        # rotation rate at radiative zone
        self.c2 = 0.2*self.ome          # latitudinal gradient of differential rotation

        ## Diffusivity
        self.etc = 1.e9
        self.ett = 1.e11

        ## Alpha effect
        self.cso = 35 # alpha non-dimensional parameter
        self.so0 = self.cso*self.ett/self.RSUN # alpha effect amplitude
        self.r1  = 0.95*self.RSUN # bottom of alpha effect
        self.d1  = 0.05*self.RSUN # width of alpha effect

        ## Meridional circulation
        self.uu0 = 1000 # flow amplitude
        self.rrb = 0.65*self.RSUN # base of the meridional flow

        # Flag for continuation
        self.cont_flag = True

        # fixed parameters (modifications are not recommended)
        self.datadir = 'data/'
        self.gridfile = self.datadir+'grid.npz'
        self.setupfile = self.datadir+'setup.npz'
        self.configfile = self.datadir+'config.json'
        
    def save(self):
        """Save configuration to a JSON file."""
        params = {k: getattr(self, k) for k in dir(self)
                    if not k.startswith("__") and not callable(getattr(self, k))}
        with open(self.configfile, 'w') as f:
            json.dump(params, f, indent=4)

    def load(self):
        """Load configuration from a JSON file."""
        with open(self.configfile, 'r') as f:
            params = json.load(f)
        for k, v in params.items():
            setattr(self, k, v)        