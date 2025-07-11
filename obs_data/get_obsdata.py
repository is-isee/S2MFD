import numpy as np
import sys
sys.path.append('../')
import S2MFD

csv_path = 'obs_data/SN_Yearly.csv'
FILENAME = 'SN_Yearly'

FOLDER = 'obs_png/'
FILENAME = FOLDER+FILENAME
data = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
years = data[:, 0]
sunspots = data[:, 1]

# 40日の補完を行い、GAに適用できるようにする
interval = 40 / 365.0 # インターバル算出（年換算）
years_interp = np.arange(years[0], years[-1], interval)
seconds_interp = years_interp * 365 * 24 * 60 * 60 # 年を秒に変換
seconds_zero = seconds_interp - seconds_interp[0] # 0秒からの経過時間に変換
sunspots_interp = np.interp(years_interp, years, sunspots) # 線形補完

S2MFD.make_graph(years_interp, sunspots_interp, 'Sunspot Number (Interpolated)', 'Year', 'Sunspot Number', FILENAME+"_interp.png")
S2MFD.make_graph(years, sunspots, 'Sunspot Number', 'Year', 'Sunspot Number', FILENAME+"_raw.png")

