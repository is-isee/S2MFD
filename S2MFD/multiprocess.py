import multiprocessing
import S2MFD

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
