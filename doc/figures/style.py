"""ドキュメント用の図の共通設定 (色・書体・保存).

色の割り当ては「その色が何の仕事をしているか」で決める。

* **順序量** (人工拡散 cs、緯度、解像度) は categorical ではなく
  **sequential (青の単一色ランプ)**。虹色は使わない。
* **符号のある量** (トロイダル磁場) は **diverging (青 ↔ 赤、中点は灰色)**。
  中点に色相を置かない。
* 系列が 2 つ以上あれば必ず凡例を出し、4 つ以下なら直接ラベルも付ける
  (識別を色だけに頼らない)。

パレットは検証済みの既定値を使っている
(`dataviz` スキルの ``references/palette.md``)。
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# --- 面と文字 (light) -------------------------------------------------------
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK2 = '#52514e'
INK3 = '#8b8a85'
GRID = '#e5e4e0'

# --- 青の sequential ランプ (順序量用) --------------------------------------
# ordinal では面に一番近い段でも 2:1 を確保する。light では step 250 以上。
BLUE = {100: '#cde2fb', 150: '#b7d3f6', 200: '#9ec5f4', 250: '#86b6ef',
        300: '#6da7ec', 350: '#5598e7', 400: '#3987e5', 450: '#2a78d6',
        500: '#256abf', 550: '#1c5cab', 600: '#184f95', 650: '#104281',
        700: '#0d366b'}
ORANGE_RAMP = ['#f8c9b3', '#f2a077', '#eb6834', '#c14f24', '#8f3a1a']


def ordinal_blues(n):
    """順序量を塗るための青 n 段 (薄い方が面に近い側、step 250 以上)."""
    steps = [250, 300, 350, 400, 450, 500, 550, 600, 650, 700]
    if n == 1:
        return [BLUE[450]]
    idx = np.linspace(0, len(steps) - 1, n).round().astype(int)
    return [BLUE[steps[i]] for i in idx]


#: 発散 colormap. 青 <-> 赤、**中点は灰色** (色相を置かない)。
DIVERGING = LinearSegmentedColormap.from_list(
    's2mfd_div',
    ['#0d366b', '#256abf', '#86b6ef', '#f0efec', '#f0a3a2', '#d03b3b',
     '#7a1f1f'])

#: 単一色の sequential colormap (符号のない量用)。
SEQUENTIAL = LinearSegmentedColormap.from_list(
    's2mfd_seq', ['#fcfcfb', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab',
                  '#0d366b'])

#: 論文値を示す線の色 (系列色とは別扱いの「参照」)
REFERENCE = '#52514e'


def apply():
    """matplotlib の既定を設定する."""
    mpl.rcParams.update({
        'figure.facecolor': SURFACE,
        'axes.facecolor': SURFACE,
        'savefig.facecolor': SURFACE,
        'text.color': INK,
        'axes.labelcolor': INK2,
        'axes.edgecolor': INK3,
        'axes.linewidth': 0.8,
        'axes.grid': True,
        'axes.axisbelow': True,
        'grid.color': GRID,
        'grid.linewidth': 0.8,
        'xtick.color': INK2,
        'ytick.color': INK2,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'axes.titleweight': 'bold',
        'legend.frameon': False,
        'legend.fontsize': 9,
        'lines.linewidth': 2.0,
        'lines.markersize': 6,
        'font.size': 10,
        'figure.dpi': 130,
        'savefig.dpi': 130,
        'savefig.bbox': 'tight',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def despine(ax):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


def label_line(ax, x, y, text, color, dx=0.0, dy=0.0, **kw):
    """線の端に直接ラベルを置く (識別を色だけに頼らないため)."""
    ax.annotate(text, xy=(x, y), xytext=(x + dx, y + dy), color=color,
                fontsize=9, fontweight='bold', va='center', **kw)
