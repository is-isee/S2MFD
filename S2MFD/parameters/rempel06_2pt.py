"""Rempel (2006) 参照モデル + 上部境界 0.985 RSUN + 2 点集中の非一様格子.

タコクライン (0.715 RSUN) と表面 (0.985 RSUN) の両方に点を集める。
一様格子だと表面で dr/Hp = 0.41 になり、v_r が dr/Hp にほぼ比例する
数値的な成分を持ってしまう (2026-08-21 の測定)。2 点集中で 0.18 まで
下げられ、これは 0.96 RSUN 領域の一様格子 (0.170) と同等。
"""
from S2MFD.parameters.rempel06 import *   # noqa: F401,F403
from S2MFD.parameters.rempel06 import RSUN

rrmax = 0.985*RSUN

grid_stretch = [2.0, 3.0]
grid_stretch_center = [0.715*RSUN, 0.985*RSUN]
grid_stretch_width = [0.05*RSUN, 0.03*RSUN]
