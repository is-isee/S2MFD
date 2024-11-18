import matplotlib.pyplot as plt
import numpy as np
import os
import S2MFD

datadir = 'data/'
data = S2MFD.S2MFD_data.initial_load(datadir)

cfg = data.cfg
grid = data.grid
setup = data.setup

fig = plt.figure('dynamo',figsize=(10,10))

n1 = 0
if os.path.isdir('./data'):
    # dataディレクトリ内の最も大きな番号を探る
    # 特定のステップから始めたい場合は、そのステップを手で指定する
    files = os.listdir('./data')
    for file in files:
        filel = file.split('.')
        if filel[0] == 'data':
            n1 = max(n1, int(filel[1]))

n0 = 0
tau_diff = data.cfg.RSUN**2/data.cfg.ett
timet = np.zeros(n1-n0)
Brrt = np.zeros((grid.ixg,grid.jxg,n1-n0))
Btht = np.zeros((grid.ixg,grid.jxg,n1-n0))
Bpht = np.zeros((grid.ixg,grid.jxg,n1-n0))
for n  in range(n0,n1):
    print(n)
    data.data_load(n)
    Brr, Bth = S2MFD.poloidal_mag(data.Aph, grid.RR, grid.sinTH, grid.drr, grid.dth)
    d = np.load(file='data/data.'+str(n).zfill(6)+'.npz')
    timet[n-n0] = d['time']
    Brrt[:,:,n-n0] = Brr
    Btht[:,:,n-n0] = Bth
    Bpht[:,:,n-n0] = d['Bph']