import S2MFD
                         
def bisection_simulation(cfg=None, parameter_file=None, datadir=None, startpoint=0):
   """
   Launches the simulation
   
   Parameters
   ----------
   cfg : S2MFD.Cfg, optional
      instance of S2MFD.Cfg
   parameter_file : str, optional
      file path to the parameter file.
      file path is relative to the S2MFD directory.
      
   Example
   -------
   If you want to run the simulation with the default parameter file, you can do:

   >>> import S2MFD
   >>> S2MFD.bisection_simulation()
   
   If you want to run the simulation with a specific parameter file, you can do:

   >>> import S2MFD
   >>> S2MFD.bisection_simulation(parameter_file = 'parameters/parameter_sample.py',datadir = 'data_sample',startpoint=500)
   
   If you want to edit parameters in the parameter file, you can do:

   >>> import S2MFD
   >>> cfg = S2MFD.Cfg('parameters/alpha_omega.py')
   >>> cfg.m = 2
   >>> S2MFD.run_simulation(cfg)
   """
   
   if datadir is None:
      print('You need to specify the datadir')
      return
   if cfg is None:   
      if parameter_file is None:
         cfg = S2MFD.Cfg()
      else:
         cfg = S2MFD.Cfg(parameter_file)
   sim = S2MFD.Simulation(cfg)
   n1, Bpht, Apht, uu0t, so0t, nt, ndt, timet = sim.load_for_bisection(datadir)
   Sunspot_N,Sunspot_N2 = sim.pre_snumbers_energy(Bpht)
   sim.initialize_simulation()
   sim.cfl_condition()
   sim.initial_for_bisection(Bpht, Apht, uu0t, so0t, nt, ndt, timet, startpoint)
   sim.bisection_ver2(uu0t=uu0t,Bpht=Bpht,Apht=Apht)
   # sim.main_loop_for_bisection(Sunspot_N=Sunspot_N,Bpht=Bpht,Apht=Apht)


