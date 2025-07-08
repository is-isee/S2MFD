import sys
sys.path.append('../')
import S2MFD
import numpy as np
import itertools
from S2MFD.parameters.defaults import *

cfg = S2MFD.Cfg()
cfg.tend = 300*365*d2s
cfg.boundary_condition_type = 'potential'

uu0_list = np.arange(900,1900,50)
so0_list = np.arange(0,80,5)


for i in range (len(uu0_list)):
    for j in range (len(so0_list)):
        cfg.uu0 = uu0_list[i]
        cfg.so0 = so0_list[j]
        cfg.datadir = f'data_initials/data_uu0={cfg.uu0}_so0={cfg.so0}/'
        print(f"uu0: {cfg.uu0}, so0: {cfg.so0}")
        S2MFD.run_simulation(cfg=cfg)
