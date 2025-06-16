import matplotlib.pyplot as plt
import numpy as np
import os, sys
sys.path.append('../')
import S2MFD
from matplotlib.animation import FuncAnimation

# TODO : 参照ファイル、動画を作成するためのディレクトリと出力ファイル名を指定
datadir = 'data/'
image_directory = 'video_data'  # PNGファイルが保存されているディレクトリ
output_video = os.path.join(image_directory, 'magnetic_field.mp4')  # 出力するMP4ファイル名

# ディレクトリが存在しない場合は作成
if not os.path.exists(image_directory):
    os.makedirs(image_directory)

data = S2MFD.Data.initial_load(datadir)

cfg = data.cfg
grid = data.grid
setup = data.setup
plt.clf()
plt.close('all')
fig = plt.figure('dynamo',figsize=(8,16))   
ax = fig.add_subplot(111,aspect='equal')

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
n1 = 1
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
    print(n)
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
    if n % 4 == 0:
        ax.clear()
        ax.pcolormesh(grid.Y/cfg.RSUN,grid.X/cfg.RSUN,d['Bph'],vmax=5.e0,vmin=-5.e0,cmap='bwr',shading='auto')
        ax.contour(grid.Y/cfg.RSUN,grid.X/cfg.RSUN,grid.RR/cfg.RSUN*grid.sinTH*d['Aph']/cfg.RSUN,colors='black',levels=np.linspace(-0.02,0.02,16))
        radius = grid.rrmax/cfg.RSUN
        ax.plot(radius*np.sin(grid.th),radius*np.cos(grid.th),color='black',alpha=0.4)
        radius = grid.rrmin/cfg.RSUN         
        ax.plot(radius*np.sin(grid.th),radius*np.cos(grid.th),color='black',alpha=0.4)
        ax.set_xlabel(r'$r/R$', fontsize=20)
        ax.set_ylabel(r'$r/R$', fontsize=20)
        ax.set_title("Evolution of Internal Magnetic Field", fontsize=25)
        ax.set_xlim( 0,1)
        ax.set_ylim(-1,1)
        plt.savefig(os.path.join(image_directory, str(n).zfill(6) + '.png'), dpi=300)
    
import os
import imageio

def create_video_from_images(image_dir, output_file, fps=10):
    """
    PNGファイルを統合してMP4動画を作成する。

    Parameters
    ----------
    image_dir : str
        PNGファイルが保存されているディレクトリのパス。
    output_file : str
        出力するMP4ファイルのパス。
    fps : int
        動画のフレームレート（1秒あたりのフレーム数）。
    """
    # ディレクトリ内のPNGファイルを取得してソート
    images = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')])
    if not images:
        print("No PNG files found in the specified directory.")
        return

    # 動画を作成
    with imageio.get_writer(output_file, fps=fps, codec="libx264") as writer:
        for image_file in images:
            print(f"Adding {image_file} to video...")
            image = imageio.imread(image_file)
            writer.append_data(image)

    print(f"Video saved as {output_file}")
create_video_from_images(image_directory, output_video, fps=10)