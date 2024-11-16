import S2MFD.config as cfg
import S2MFD
import importlib

importlib.reload(cfg)
S2MFD.run_simulation(cfg)