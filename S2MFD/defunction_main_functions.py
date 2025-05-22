import os
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
   from concurrent.futures import ProcessPoolExecutor

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
   Sunspot_N, Sunspot_N2 = sim.pre_snumbers_energy(Bpht)
   # ============================================================================= #
   
   # ============================================================================= #
   # validate the simulation
   A = [9, 10, 11]
   Omg = [1/694762000, 1/694792000]
   B = [30, 32, 35]
   C = [np.pi/6, np.pi/4, 0]
   # ============================================================================= # 
   
   # ============================================================================= #
   # Multiprocessing
   with ProcessPoolExecutor() as executor:
      futures = []
      for idx, (A_sample, omg_sample, B_sample, C_sample) in enumerate(product(A, Omg, B, C)):
         # 一意のディレクトリ名を生成
         result_dir = f"data_defunction/data{idx:04d}/"

         # タスクを実行
         futures.append(executor.submit(
               run_defunction, parameter_file, A_sample, omg_sample, B_sample, C_sample, result_dir, n1, Bpht, Apht, uu0t, so0t, nt, ndt, timet, startpoint, endpoint, Sunspot_N
            ))

      for future in futures:
         if future.result():  # シミュレーションが終了した場合
            executor.shutdown(wait=False)
            print(idx)
            break


def run_defunction(parameter_file, A_sample, omg_sample, B_sample, C_sample, result_dir, n1, Bpht, Apht, uu0t, so0t, nt, ndt, timet, startpoint, endpoint, Sunspot_N):
   """
   実行するシミュレーション
   """
   cfg = S2MFD.Cfg(parameter_file)
   cfg.datadir = result_dir  # 結果を保存するディレクトリを設定
   sim = S2MFD.Simulation(cfg)
   sim.initialize_simulation()
   sim.cfl_condition()
   print('A_sample=', A_sample)
   print('omg_sample=', omg_sample)
   print('B_sample=', B_sample)
   print('Result directory:', result_dir)
   # ============================================================================= #
   # First time step is gotten from the reference data
   sim.initial_for_defunction(Bpht=Bpht, Apht=Apht, uu0t=uu0t, so0t=so0t, nt=nt, ndt=ndt, timet=timet, index=startpoint, index_end=endpoint)
   # ============================================================================= #
   sim.defunction_main_loop(A_sample=A_sample, omg_sample=omg_sample, B_sample=B_sample, C_sample=C_sample, timet=timet, index_start=startpoint, index_end=endpoint)
   judge = sim.judge(Sunspot_N[startpoint:endpoint+1])
   if judge == 1:
      print('The simulation is finished')
      print("A=", A_sample, "Omg=", omg_sample, "B=", B_sample)
      oldpath = result_dir
      newpath = f"data_defunction/data_true/"
      os.rename(oldpath, newpath)
      return True
   else:
      print('The simulation is not finished')
      return False