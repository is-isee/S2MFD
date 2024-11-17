import matplotlib.pyplot as plt
import numpy as np
import os
import S2MFD

datadir = 'data/'
data = S2MFD.S2MFD_data.initial_load(datadir)

grid = data.grid

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
time = np.zeros(n1-n0)
Brrt = np.zeros((ixg,jxg,n1-n0))
Btht = np.zeros((ixg,jxg,n1-n0))
Bpht = np.zeros((ixg,jxg,n1-n0))
for n  in range(n0,n1):
    print(n)
    d = np.load(file='data/data.'+str(n).zfill(6)+'.npz')
    time[n-n0] = d['time']
    Brrt[:,:,n-n0] = d['Brr']
    Btht[:,:,n-n0] = d['Bth']
    Bpht[:,:,n-n0] = d['Bph']