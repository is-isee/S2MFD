import sys
sys.path.append('../')
import S2MFD
import numpy as np
import itertools
from S2MFD.parameters.defaults import *
import matplotlib.pyplot as plt
import os
import concurrent.futures


uu0_list = np.arange(600,800,20)

def run_and_analyze(uu0):
    cfg = S2MFD.Cfg()
    cfg.tend = 110*365*d2s
    cfg.boundary_condition_type = 'potential'
    cfg.uu0 = uu0
    cfg.so0 = 50
    cfg.datadir = f'data_initials/data_uu0={cfg.uu0}_so0={cfg.so0}/'
    print(f"uu0: {cfg.uu0}, so0: {cfg.so0}")
    S2MFD.run_simulation(cfg=cfg)

    datadir = cfg.datadir
    data = S2MFD.Data.initial_load(datadir)

    cfg = data.cfg
    grid = data.grid
    setup = data.setup

    fig = plt.figure('dynamo',figsize=(10,10))

    n1 = 0
    if os.path.isdir(datadir):
        # dataディレクトリ内の最も大きな番号を探る
        # 特定のステップから始めたい場合は、そのステップを手で指定する
        files = os.listdir(datadir)
        for file in files:
            filel = file.split('.')
            if filel[0] == 'data':
                n1 = max(n1, int(filel[1]))

    n0 = 0
    tau_diff = data.cfg.RSUN**2/data.cfg.ett
    timet = np.zeros(n1-n0)
    nt = np.zeros(n1-n0)
    ndt = np.zeros(n1-n0)
    Brrt = np.zeros((grid.ixg,grid.jxg,n1-n0))
    Btht = np.zeros((grid.ixg,grid.jxg,n1-n0))
    Bpht = np.zeros((grid.ixg,grid.jxg,n1-n0))
    Apht = np.zeros((grid.ixg,grid.jxg,n1-n0))
    so0t = np.zeros(n1-n0)
    uu0t = np.zeros(n1-n0)
    dltt = np.zeros(n1-n0)

    for n  in range(n0,n1):
        data.data_load(n)
        Brr, Bth = S2MFD.physics.poloidal_mag(data.Aph, grid.RR, grid.sinTH, grid.drr, grid.dth)
        d = np.load(file=datadir+'data.'+str(n).zfill(6)+'.npz')
        timet[n-n0] = d['time']
        nt[n-n0] = d['n']
        ndt[n-n0] = d['nd']
        Brrt[:,:,n-n0] = Brr
        Btht[:,:,n-n0] = Bth
        Bpht[:,:,n-n0] = d['Bph']
        Apht[:,:,n-n0] = d['Aph']
        so0t[n-n0] = d['so0']
        uu0t[n-n0] = d['uu0']
        if 'dl' in d:
            dltt[n-n0] = d['dl']

    Bpht0 = Bpht[1+np.argmin(abs(grid.rr-0.7*cfg.RSUN)),np.argmin(abs(grid.th-30/180*np.pi)),:]
    Brrt0 = Brrt[-2,np.argmin(abs(grid.th-60/180*np.pi)),:]

    Bpht0_sign = np.sign(Bpht0)
    Bpht0_sign_diff = np.diff(Bpht0_sign)

    Brrt0_sign = np.sign(Brrt0)
    Brrt0_sign_diff = np.diff(Brrt0_sign)
    ne_r = np.where(Brrt0_sign_diff == +2)[0][-1]


    ns = np.where(Bpht0_sign_diff == +2)[0][-2]
    ne = np.where(Bpht0_sign_diff == +2)[0][-1]
    timeu = (timet[ns:ne]-timet[ns])/tau_diff
    timeur = (timet[ns:ne_r]-timet[ns])/tau_diff
    Bpht0u = Bpht0[ns:ne]
    Brrt0u = Brrt0[ns:ne]
    nw = ne - ns
    nm = np.argmax(Bpht0u)
    period_year = timeu[-1]*tau_diff/86400/365
    print('Period(year) = ', period_year, 'year')

    csv_file = "uu0_period.csv"
    write_header = not os.path.exists(csv_file)

    with open(csv_file, "a") as f:
        if write_header:
            f.write("uu0,period_year\n")
        f.write(f"{uu0},{period_year}\n")

# 並列実行
with concurrent.futures.ProcessPoolExecutor() as executor:
    list(executor.map(run_and_analyze, uu0_list))

