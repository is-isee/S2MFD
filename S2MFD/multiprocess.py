import multiprocessing
import S2MFD
import os
import numpy as np
# =========================================================================================================== #
# 1次元の並列化
def survey_simulation_worker(ii, cfg = None, parameter_file = None):
    so0_list = [5.0, 10.0, 30.0, 50.0, 70.0]
    uu0_list = [250, 500, 750, 1000, 1500]
    
    if cfg is None:
        if parameter_file is None:
            cfg = S2MFD.Cfg()
        else:
            cfg = S2MFD.Cfg(parameter_file)
    
    cfg.so0 = so0_list[ii]
    for jj in range(len(uu0_list)):
        print(ii,jj)
        cfg.uu0 = uu0_list[jj]
        cfg.datadir = f"data_{ii}{jj}/"
        sim = S2MFD.Simulation(cfg)
        sim.initialize_simulation()
        sim.cfl_condition()
        sim.initial_condition()
        sim.main_loop()

def main_multi(cfg=None, parameter_file=None):
    num_processes = np.minimum(os.cpu_count(), 5) # 32まで並列化可能

    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.starmap(survey_simulation_worker, [(ii, cfg, parameter_file) for ii in range(num_processes)])
# =========================================================================================================== #



# =========================================================================================================== #
# 全並列化
def survey_simulation_worker_all(ii, jj, cfg=None, parameter_file=None):
    so0_list = [5.0, 10.0, 30.0, 50.0, 70.0]
    uu0_list = [250, 500, 750, 1000, 1500]
    
    if cfg is None:
        if parameter_file is None:
            cfg = S2MFD.Cfg()
        else:
            cfg = S2MFD.Cfg(parameter_file)
    
    cfg.so0 = so0_list[ii]
    cfg.uu0 = uu0_list[jj]
    cfg.datadir = f"data_{ii}{jj}/"
    
    print(f"Running simulation for ii={ii}, jj={jj}")
    
    sim = S2MFD.Simulation(cfg)
    sim.initialize_simulation()
    sim.cfl_condition()
    sim.initial_condition()
    sim.main_loop()

def main_multi_all(cfg=None, parameter_file=None):
    so0_list = [5.0, 10.0, 30.0, 50.0, 70.0]
    uu0_list = [250, 500, 750, 1000, 1500]
    num_processes = min(os.cpu_count(), len(so0_list)*len(uu0_list))

    task_list = [(ii, jj, cfg, parameter_file) for ii in range(len(so0_list)) for jj in range(len(uu0_list))]

    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.starmap(survey_simulation_worker_all, task_list)
# =========================================================================================================== #