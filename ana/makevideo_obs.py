import matplotlib.pyplot as plt
import numpy as np
import os, sys
sys.path.append('../')
import S2MFD

# 分析範囲、対象は手で決める
image_directory = 'video'  # PNGファイルが保存されているディレクトリ
n0 = 2226
n1 = 2702
def initial_load(i,datadir):
    """ 
    Load initial configuration, grid, and setup from the specified directory.

    Parameters
    ----------
    datadir : str
        Directory containing the configuration, grid, and setup files.

    Returns
    -------
    S2MFD.S2MFD_data
        An instance of the S2MFD_data class.
    """        
    cfg = S2MFD.Cfg.load(datadir+str(i)+'config.json')
    cfg.datadir = datadir
    grid = S2MFD.Grid.load(datadir+str(i)+cfg.gridfile)
    setup = S2MFD.Setup.load(datadir+str(i)+cfg.setupfile)
    legendre = S2MFD.Legendre.load(datadir+str(i)+cfg.legendrefile)
    
    return cfg,grid,setup

def get_data_file_path(datadir,i,nd):
    """
    Generate the file path for a specific step.

    Parameters
    ----------
    nd : int
        Data output step 

    Returns
    -------
    str
        File path for the specified data step.
    """        
    return datadir+str(i)+'data.'+str(nd).zfill(6)+'.npz'

def data_load(datadir,i,nd):
    """
    Load data for `Bph`, `Aph`, `time`, `n` and `nd`, from a file for a specific step.

    Parameters
    ----------
    nd : int
        Data output step 
    """        
    filename = get_data_file_path(datadir,i,nd)
    d = np.load(file=filename, allow_pickle=True)
    Bph = d['Bph']
    Aph = d['Aph']
    time = d['time']
    n = d['n']
    nd = nd
    return Bph, Aph, time, n, nd
# ============================================================================== #
# 観測データの読み込み
data = np.genfromtxt("../obs_data/obs_data/SN_Yearly_interp.csv", delimiter=',', skip_header=1)
time1 = data[n0:n1, 1]/24/60/60/365
SN1 = data[n0:n1, 3]
# ============================================================================== #
# datadir2
for i in range(0,42):
    datadir = '../datavideo/data_18/'+str(i)+'/'
    cfg,grid,setup = initial_load(i,datadir)

    # ============================================================================== #
    # 黒点などを割り出すための関数
    base  = 1+np.argmin(abs(grid.rr-0.7*cfg.RSUN))
    # 移動平均関数
    def moving_average(SunspotsNum, n_conv):
        conv_f = np.ones(n_conv)/n_conv
        SN_smooth = np.convolve(SunspotsNum, conv_f, mode='same')#移動平均
        
        return SN_smooth
    # 磁場エネルギー密度B^2(toroidal,15°,r_c)を黒点数の指標とする。
    def Karak_deffine(Bpht,base,cfg,grid):
        gamma = 5.8653520852  # 観測に基づいた調整パラメタ
        SN = gamma*Bpht[base,np.argmin(abs(grid.th-75/180*np.pi)),:]**2
        SN2 = moving_average(SN, 4)
        return SN

    # 磁場エネルギー密度B^2(toroidal,10°~20°,r_c)を黒点数の指標とする。
    def original_deffine(Bpht,base,cfg,grid):
        locap =   np.argmin(abs(grid.th- 80/180*np.pi))
        locam =   np.argmin(abs(grid.th- 70/180*np.pi))
        SN = np.mean(Bpht[base,locam:locap,:]**2,axis=0)
        SN2 = moving_average(SN, 4)
        return SN2

    fig = plt.figure('dynamo',figsize=(10,10))
    tau_diff = cfg.RSUN**2/cfg.ett
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

    for nn in range(n0,n1):
        print(nn)
        Bph, Aph, time, n, nd = data_load(datadir,i,nn)
        Brr, Bth = S2MFD.physics.poloidal_mag(Aph, grid.RR, grid.sinTH, grid.drr, grid.dth)
        d = np.load(file=datadir+str(i)+'data.'+str(nn).zfill(6)+'.npz')
        timet[nn-n0] = d['time']
        nt[nn-n0] = d['n']
        ndt[nn-n0] = d['nd']
        Brrt[:,:,nn-n0] = Brr
        Btht[:,:,nn-n0] = Bth
        Bpht[:,:,nn-n0] = d['Bph']
        Apht[:,:,nn-n0] = d['Aph']
        so0t[nn-n0] = d['so0']
        uu0t[nn-n0] = d['uu0']
        if 'dl' in d:
            dltt[nn-n0] = d['dl']
        
    SN2 = Karak_deffine(Bpht,base,cfg,grid)
    so0t2= so0t
    uu0t2 = uu0t
    time2 = timet/cfg.d2s/365
    # ============================================================================== #
    # 基本ラベルサイズの指定
    size = 10

    # 日本語使用可能
    import matplotlib as mpl
    mpl.rcParams['font.family'] = 'IPAPGothic'
    cc   = np.sum((SN1-np.mean(SN1))*(SN2-np.mean(SN2)))/np.sqrt(np.sum((SN1-np.mean(SN1))**2)*np.sum((SN2-np.mean(SN2))**2))
    # ============================================================================== #
    # 黒点数の比較グラフ
    # グラフの描画
    plt.figure(figsize=(10, 6))  # グラフのサイズを調整
    plt.plot(time1, SN1, 'r--', label='観測',linewidth=2.5)
    plt.plot(time2, SN2, 'b', label='推定')

    # 相関係数をグラフ左上に表示
    plt.text(0.05, 0.95, f'相関係数 = {cc:.3f}', 
            transform=plt.gca().transAxes,  # 軸の座標系(0〜1)で指定
            fontsize=12, 
            verticalalignment='top')
    # 軸ラベル・タイトル
    plt.title('黒点数時間変化', fontsize=4*size)  # タイトルを追加
    plt.xlabel('年', fontsize=3*size)
    plt.ylabel('黒点相対数', fontsize=3*size)

    # 軸のメモリフォントサイズ
    plt.xticks(fontsize=2*size)  # x軸の数値サイズを調整
    plt.yticks(fontsize=2*size)  # y軸の数値サイズを調整

    # グリッドを追加して見やすく
    plt.grid(True, linestyle='--', alpha=0.7)
    # 凡例を表示
    plt.legend(fontsize=1.6*size, loc='upper right')  # 凡例を右上に固定
    plt.tight_layout()
    # グラフを保存
    plt.savefig(os.path.join(image_directory, str(i).zfill(6) + '.png'), dpi=300)
    plt.clf()

"""
# ============================================================================== #
# u0の比較グラフ
plt.plot(time2,uu0t2,'b',label='推定')
plt.xlabel('年',fontsize=3*size)
plt.ylabel(r'$u_0(\rm{cm/s})$',fontsize=3*size)
plt.title(r'$u_0$ 時間変化', fontsize=4*size)  # タイトルを追加
ymax = max(np.max(uu0t2), np.max(uu0t2)) * 1.1
plt.ylim(bottom=0, top=ymax)
# 軸のメモリを細かく設定
plt.xticks(fontsize=2*size)  # x軸の数値サイズを調整
plt.yticks(fontsize=2*size)  # y軸の数値サイズを調整
# グリッドを追加して見やすく
plt.grid(True, linestyle='--', alpha=0.7)
# 凡例を表示
plt.legend(fontsize=1.6*size, loc='upper right')  # 凡例を右上に固定
# グラフを保存
plt.savefig("P_u0_compare.png", dpi=300)
plt.clf()

# ============================================================================== #
# so0の比較グラフ
plt.plot(time2,so0t2,'b',label='推定')
ymax = max(np.max(so0t2), np.max(so0t2)) * 1.1
plt.ylim(bottom=0, top=ymax)
plt.xlabel('年',fontsize=3*size)
plt.ylabel(r'$s_0(\rm{cm/s})$',fontsize=3*size)
plt.title(r'$s_0$ 時間変化', fontsize=4*size)  # タイトルを追加
# 軸のメモリを細かく設定
plt.xticks(fontsize=2*size)  # x軸の数値サイズを調整
plt.yticks(fontsize=2*size)  # y軸の数値サイズを調整
# グリッドを追加して見やすく
plt.grid(True, linestyle='--', alpha=0.7)
# 凡例を表示
plt.legend(fontsize=1.6*size, loc='upper right')  # 凡例を右上に固定
# グラフを保存
plt.savefig("P_s0_compare.png", dpi=300)
plt.clf()
"""

