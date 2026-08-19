"""ラン一式を読み込んで対話解析の名前空間を用意するスクリプト。

使い方:
    python ana.py [datadir]         # 既定は ../data/
または IPython で:
    %run ana.py ../data_xxx/
"""
import matplotlib.pyplot as plt
import numpy as np

from ana_common import get_datadir, load_run

datadir = get_datadir()
run = load_run(datadir)

# 対話利用のために変数を展開しておく (従来のスクリプト互換)
data = run.data
cfg = run.cfg
grid = run.grid
setup = run.setup
timet = run.timet
Bpht = run.Bpht
Apht = run.Apht
Brrt = run.Brrt
Btht = run.Btht
tau_diff = run.tau_diff
n0, n1 = run.n0, run.n1

print(f'loaded {datadir}: n={n0}..{n1}')
