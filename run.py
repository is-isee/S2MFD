import S2MFD
import sys

cfg = S2MFD.config_c('parameters/alpha_omega_etaconst.py')
cfg.datadir = 'data_alpha_omega_etaconst/'
S2MFD.run_simulation(cfg)

cfg = S2MFD.config_c('parameters/alpha_omega.py')
cfg.datadir = 'data_alpha_omega/'
S2MFD.run_simulation(cfg)

cfg = S2MFD.config_c('parameters/defaults.py')
cfg.datadir = 'data_flux_transport/'
S2MFD.run_simulation(cfg)

#S2MFD.run_simulation(parameter_file='parameters/alpha_omega.py')