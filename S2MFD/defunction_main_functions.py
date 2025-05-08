import S2MFD
                         
def defunction_simulation(cfg=None, parameter_file=None, datadir=None, startpoint=0, endpoint=0):
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
   from itertools import product
   import numpy as np
   
   if datadir is None:
      print('You need to specify the datadir')
      return
   if cfg is None:   
      if parameter_file is None:
         cfg = S2MFD.Cfg()
      else:
         cfg = S2MFD.Cfg(parameter_file)
   # ============================================================================= #
   # Load the parameter file
   sim = S2MFD.Simulation(cfg)
   n1, Bpht, Apht, uu0t, so0t, nt, ndt, timet = sim.load_for_bisection(datadir)
   Sunspot_N,Sunspot_N2 = sim.pre_snumbers_energy(Bpht)
   # ============================================================================= #
   
   # ============================================================================= #
   # No change
   sim.initialize_simulation()
   sim.cfl_condition()
   # ============================================================================= #   
   A = [10, 12.0]
   Omg = [693792000, 693793000]
   B = [35]
   C = [0, np.pi/6, np.pi/3, np.pi/2, np.pi, 2*np.pi/3, 5*np.pi/6]
   # A = [10]
   # Omg = [693792000]
   # B = [35]
   
   for A_sample, omg_sample, B_sample, C_sample in product(A, Omg, B, C):
      print('A_sample=', A_sample)
      print('omg_sample=', omg_sample)
      print('B_sample=', B_sample)
      # ============================================================================= #
      # First time step is gotten from the reference data
      sim.initial_for_defunction(Bpht=Bpht, Apht=Apht, uu0t=uu0t, so0t=so0t, nt=nt, ndt=ndt, timet=timet, index=startpoint, index_end=endpoint)
      # ============================================================================= #
      sim.defunction_main_loop(A_sample=A_sample,omg_sample=omg_sample,B_sample=B_sample,C_sample=C_sample,timet=timet,index_start=startpoint,index_end=endpoint)
      judge = sim.judge(Sunspot_N[startpoint:endpoint+1])
      if judge == 1:
         print('The simulation is finished')
         print("A=",A_sample,"Omg=",omg_sample,"B=",B_sample)
         break

