"""スナップショットの磁場を人為的に加工するスクリプト。

南北非対称性を導入する等の数値実験用。
既定では元ファイルを上書きせず `*_edited.npz` に保存する
(旧実装は元ファイルを破壊的に上書きしていた)。

使い方:
    python npz_edit.py [datadir] [nd]
"""
import os
import sys

import numpy as np

import S2MFD

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ana_common import radial_index

datadir = sys.argv[1] if len(sys.argv) > 1 else '../data/'
nd = int(sys.argv[2]) if len(sys.argv) > 2 else 800

# .npz ファイルを開く
srcfile = os.path.join(datadir, f'data.{nd:06d}.npz')
d = np.load(srcfile)

data = S2MFD.Data.initial_load(datadir)

cfg = data.cfg
grid = data.grid
setup = data.setup

# 各データを展開
Bph = d["Bph"]
Aph = d["Aph"]
time = d["time"]
n = d["n"]


# ゴーストを踏まない添字 (2026-08-23)。以前の 1+argmin(...) は
# 1 セル外側を指していた。ana_common 参照。
base  = radial_index(grid, 0.8*cfg.RSUN)
# グリッドに応じた北半球と南半球の分割
if grid.jxg % 2==0:
    S_equa = grid.jxg//2 - 1
    N_equa = S_equa + 1
else:
    S_equa = (grid.jxg-1)//2 - 1
    N_equa = S_equa + 2

gamma_s = 0.4
gamma_n = 0.0
gamma_t = 0.8
Aph[base:,:S_equa] = gamma_s * Aph[base:,:S_equa]
Aph[base:,N_equa:] = gamma_n * Aph[base:,N_equa:]
Bph[    :,      :] = gamma_t * Bph[    :,      :]

# 修正後のデータを別ファイルに保存 (上書きしたい場合は手動でリネームする)
outfile = os.path.join(datadir, f'data.{nd:06d}_edited.npz')
np.savez(outfile, Bph=Bph, Aph=Aph, time=time, n=n)
print(f'wrote {outfile}')
