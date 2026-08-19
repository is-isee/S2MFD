import S2MFD
                              
def run_simulation(cfg=None, parameter_file=None):
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
   >>> S2MFD.run_simulation()
   
   If you want to run the simulation with a specific parameter file, you can do:

   >>> import S2MFD
   >>> S2MFD.run_simulation(parameter_file='parameters/alpha_omega.py')
   
   If you want to edit parameters in the parameter file, you can do:

   >>> import S2MFD
   >>> cfg = S2MFD.Cfg('parameters/hotta10.py')
   >>> cfg.m = 2   # 基本量の変更は resolve() で派生量 (c1d, c2d) に反映される
   >>> S2MFD.run_simulation(cfg)

   Returns
   -------
   S2MFD.Simulation
      The simulation object after the run (fields, time, etc.).
   """
   if cfg is None:
      if parameter_file is None:
         cfg = S2MFD.Cfg()
      else:
         cfg = S2MFD.Cfg(parameter_file)

   sim = S2MFD.Simulation(cfg)
   sim.initialize_simulation()
   sim.cfl_condition()
   sim.initial_condition()
   sim.main_loop()

   return sim
