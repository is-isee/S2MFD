"""Rempel (2006) 表 1 と計算結果を突き合わせる。

表 1 の量 (すべて magnetic buoyancy ON の列):

           Quantity              REF   a=0.125  a=0.25  a=0.5
  (Om_eq - Om_pole)/Om0          0.27    0.21    0.15    0.10
  max(Om - Om_bar) 90deg [nHz]     -      4.7    11.5    17.7
  max(Om - Om_bar) 60deg [nHz]     -      3.5     6.9     8.3
  max(B_phi) [T]                   -      1.2     1.4     1.2
  max(B_r)   [T]                   -    0.014   0.022   0.027
  Q_Lambda [F_sun]               0.014   0.013   0.011   0.008
  Q_nu^Omega / Q_Lambda          0.574   0.5     0.405   0.304
  Q_C / Q_Lambda                 0.425   0.43    0.445   0.465
  Q_L^Omega / Q_Lambda             -     0.069   0.149   0.232
  Q_B / Q_Lambda                 0.419   0.41    0.414   0.423
  Q_nu^M / Q_Lambda              0.005   0.005   0.007   0.010
  Q_L^M / Q_Lambda                 -     0.014   0.024   0.033
  Q_eta / Q_Lambda                 -     0.083   0.172   0.268
  E_B_bar [1e31 J]                 -      2.8     4.6     5.1
  max[(E_B - E_B_bar)/E_B_bar]     -     0.12    0.22    0.26

max(Om - Om_bar) と max(B_r) は 0.985 RSUN、max(B_phi) は 0.735 RSUN で評価。
Om_bar は時間平均 (dynamo7.py が指数移動平均で作る)。

E_B_bar の桁について
--------------------
表 1 の該当行の指数は読み取りにくいが **10^31 J** である。10^33 J だと
同じ表の max(E_B)_bc = 1.7e31 J (r = 0.71-0.76 RSUN の殻での値) が全体の
0.6 パーセントになり、「トロイダル磁場は対流層底に集中する」という
この模型の描像と矛盾する (殻の体積比は 12 パーセント)。
max(B_phi) = 1.2-1.4 T と E_B = int B^2/(2 mu_0) dV からの見積もりも
~3e31 J で、10^33 J には B_rms ~ 3 T が必要になり max(B_phi) を超える。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from S2MFD.physics.energy import solar_luminosity   # 3.828e33 erg/s

np.seterr(all='ignore')
YR = 3.156e7

PAPER = {                      # alpha0 [cm/s] -> 表 1 の列
    12.5: dict(DR=0.21, t90=4.7,  t60=3.5, Bph=1.2, Br=0.014,
               QL=0.013, Qnu=0.5,   QC=0.43,  QLO=0.069, QB=0.41,
               QnuM=0.005, QLM=0.014, Qeta=0.083),
    25.0: dict(DR=0.15, t90=11.5, t60=6.9, Bph=1.4, Br=0.022,
               QL=0.011, Qnu=0.405, QC=0.445, QLO=0.149, QB=0.414,
               QnuM=0.007, QLM=0.024, Qeta=0.172),
    50.0: dict(DR=0.10, t90=17.7, t60=8.3, Bph=1.2, Br=0.027,
               QL=0.008, Qnu=0.304, QC=0.465, QLO=0.232, QB=0.423,
               QnuM=0.010, QLM=0.033, Qeta=0.268),
}
for _a, _eb, _var in ((12.5, 2.8, 0.12), (25.0, 4.6, 0.22), (50.0, 5.1, 0.26)):
    PAPER[_a]['EB'] = _eb
    PAPER[_a]['EBvar'] = _var
KEYMAP = [('Q_Lambda','QL'), ('Q_nu_Omega','Qnu'), ('Q_C','QC'),
          ('Q_L_Omega','QLO'), ('Q_B','QB'), ('Q_nu_M','QnuM'),
          ('Q_L_M','QLM'), ('Q_eta','Qeta')]

def cycle_period(t, b):
    """符号反転から周期を出す (立ち上がりを除いた後半のみ)。

    **「後半」は時刻で切る。インデックスの中点ではない。**
    延長ランをつなぐと区間ごとに出力間隔が違う (42 年に 2000 点 +
    18 年に 2000 点) ので、インデックスの中点は t = 42 年になり、
    後半に反転が 2 回しか入らず周期が出せなかった (2026-08-24)。
    """
    late = t >= 0.5*(t[0] + t[-1])
    tt, bb = t[late], b[late]
    s = np.sign(bb); idx = np.where(np.diff(s) != 0)[0]
    if len(idx) < 3: return np.nan, 0
    per = 2*np.diff(tt[idx])
    return float(np.mean(per)), len(per)

def detrend(t, x, win):
    """幅 win の移動平均を引いて振動成分だけ残す (端は捨てる)。

    dynamo7.py の om_bar は時定数 1 サイクルの指数移動平均なので、解がまだ
    緩和している間はドリフトに追従しきれず、その分が「振動振幅」に乗る。
    実測 (p_a125) では生の max が 20.9 nHz、ドリフトを引くと 6.1 nHz
    (論文 4.7)。表 1 の量は時間平均まわりの振幅なので後者が対応する。
    """
    n = max(3, int(win/(t[1] - t[0])))
    m = np.convolve(x, np.ones(n)/n, mode='same')
    e = n//2
    return (x - m)[e:-e] if len(x) > 2*e + 2 else x - m

def qcols(d, h):
    """QKEYS の列を名前で引く。

    **列の位置を `h.shape[1] - len(qkeys)` で当てにしてはいけない。**
    2026-08-23 に dynamo7.py が E_Omega/E_M/放射層の磁束を末尾に足したため、
    新しい記録 (22 列) では QKEYS は 10-17 で、差し引きの 14 は的外れになる。
    実害: res144_* の Q_Lambda が 100 分の 1、Q_nu^M が 1e12 倍に見えた。
    hist_cols があればそれを使い、無い古い記録 (16 列 = 診断 8 + QKEYS 8)
    だけ差し引きにする。
    """
    q = [str(k) for k in d["qkeys"]]
    if 'hist_cols' in d:
        idx = {str(k): i for i, k in enumerate(d['hist_cols'])}
        return {k: h[:, idx[str(k)]] for k in q}
    off = h.shape[1] - len(q)
    return {k: h[:, off + i] for i, k in enumerate(q)}


def load_run(tag):
    """1 本または ``+`` でつないだ複数本の記録を読む。

    延長ラン (``res144_kin`` -> ``res144_kin_b``) は別ファイルなので、
    ``'res144_kin+res144_kin_b'`` と書けば時刻順につないで 1 本として扱う。
    重複する時刻は落とす。QKEYS は**名前で**取り出してからつなぐので、
    区間ごとに列構成が違っていても正しく合わさる。
    """
    hs, bus, qcs, th = [], [], [], None
    for tg in tag.split('+'):
        d = np.load(f'results_rempel/{tg}/dynamo.npz')
        h = d['hist']
        hs.append(h)
        bus.append(d['butter'])
        qcs.append(qcols(d, h))
        th = d['th']
    h = np.vstack(hs)
    bu = np.vstack(bus)
    qc = {k: np.concatenate([c[k] for c in qcs]) for k in qcs[0]}
    order = np.argsort(h[:, 0])
    h, bu = h[order], bu[order]
    qc = {k: v[order] for k, v in qc.items()}
    keep = np.concatenate(([True], np.diff(h[:, 0]) > 0))
    return h[keep], bu[keep], th, {k: v[keep] for k, v in qc.items()}


def analyse(tag, steady_yr=20.0):
    """末尾 steady_yr 年 (統計的定常部) だけを使って表 1 の量を出す。"""
    h, bu, th, qc = load_run(tag)
    t = h[:,0]/YR
    late = t >= t[-1] - steady_yr
    lat40 = np.argmin(abs(th - np.radians(50)))
    per, npc = cycle_period(t, bu[:, lat40])
    out = dict(t_end=t[-1], period=per, ncyc=npc,
               DR=float(np.mean(h[late,4])),
               t90=float(np.abs(detrend(t[late], h[late,5], 18.0)).max()),
               t60=float(np.abs(detrend(t[late], h[late,6], 18.0)).max()),
               Bph=float(np.abs(h[late,1]).max()),
               Br=float(np.abs(h[late,2]).max()))
    # 磁気エネルギー [1e31 J]。erg -> J は 1e-7。
    # **変動幅はドリフトを引いてから測る。** 引かないと、まだ成長している
    # 解では成長分が「サイクル変動」に乗る (実測: v2_a125 で 0.51 対 0.12)。
    eb = h[late, 7]*1e-7
    out['EB'] = float(np.mean(eb))*1e-31
    ebd = detrend(t[late], eb, 18.0)
    out['EBvar'] = float(np.abs(ebd).max()/np.mean(eb))
    out['EBvar_raw'] = float(np.abs(eb - np.mean(eb)).max()/np.mean(eb))
    ql = float(np.mean(qc['Q_Lambda'][late]))
    # 太陽光度は energy.py と揃える (以前ここだけ 3.846e33 だった)
    out['QL_Fsun'] = ql/solar_luminosity
    for name, key in KEYMAP[1:]:
        out[key] = float(np.mean(qc[name][late]))/ql
    return out

def show(tag, alpha0):
    try: r = analyse(tag)
    except Exception as e:
        print(f"{tag}: 読めない ({e})"); return
    p = PAPER.get(alpha0, {})
    print(f"\n=== {tag}  alpha0={alpha0} cm/s ({alpha0/100:.3f} m/s)  "
          f"t={r['t_end']:.1f}yr  周期 {r['period']:.1f}yr (n={r['ncyc']})")
    rows = [('(Om_eq-Om_pole)/Om0','DR','DR','%.3f'),
            ('max(Om-Om_bar) 90deg [nHz]','t90','t90','%.1f'),
            ('max(Om-Om_bar) 60deg [nHz]','t60','t60','%.1f'),
            ('max(B_phi) [T]','Bph','Bph','%.3f'),
            ('max(B_r) [T]','Br','Br','%.4f'),
            ('Q_Lambda [F_sun]','QL_Fsun','QL','%.4f'),
            ('Q_nu^Om / Q_Lam','Qnu','Qnu','%.3f'),
            ('Q_C / Q_Lam','QC','QC','%.3f'),
            ('Q_L^Om / Q_Lam','QLO','QLO','%.3f'),
            ('Q_B / Q_Lam','QB','QB','%.3f'),
            ('Q_nu^M / Q_Lam','QnuM','QnuM','%.3f'),
            ('Q_L^M / Q_Lam','QLM','QLM','%.3f'),
            ('Q_eta / Q_Lam','Qeta','Qeta','%.3f'),
            ('E_B_bar [1e31 J]','EB','EB','%.2f'),
            ('max|E_B-E_B_bar|/E_B_bar','EBvar','EBvar','%.3f'),
            ('  (ドリフト込みの生値)','EBvar_raw',None,'%.3f')]
    print(f"  {'量':<28}{'計算':>10}{'論文':>10}   比")
    for label, rk, pk, fmt in rows:
        v = r.get(rk); pv = p.get(pk) if pk else None
        if v is None: continue
        ratio = f"{v/pv:6.2f}" if pv else "     -"
        pvs = (fmt % pv) if pv is not None else "-"
        print(f"  {label:<28}{fmt % v:>10}{pvs:>10}   {ratio}")

if __name__ == '__main__':
    for spec in sys.argv[1:]:
        tag, a = spec.split(':')
        show(tag, float(a))
