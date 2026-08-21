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

max(Om - Om_bar) と max(B_r) は 0.985 RSUN、max(B_phi) は 0.735 RSUN で評価。
Om_bar は時間平均 (dynamo7.py が指数移動平均で作る)。
"""
import sys, numpy as np
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
KEYMAP = [('Q_Lambda','QL'), ('Q_nu_Omega','Qnu'), ('Q_C','QC'),
          ('Q_L_Omega','QLO'), ('Q_B','QB'), ('Q_nu_M','QnuM'),
          ('Q_L_M','QLM'), ('Q_eta','Qeta')]

def cycle_period(t, b):
    """符号反転から周期を出す (立ち上がりを除いた後半のみ)。"""
    half = len(t)//2
    tt, bb = t[half:], b[half:]
    s = np.sign(bb); idx = np.where(np.diff(s) != 0)[0]
    if len(idx) < 3: return np.nan, 0
    per = 2*np.diff(tt[idx])
    return float(np.mean(per)), len(per)

def analyse(tag):
    d = np.load(f'results_rempel/{tag}/dynamo.npz')
    h = d['hist']; q = list(d['qkeys']); th = d['th']; bu = d['butter']
    t = h[:,0]/YR
    late = slice(len(h)//2, None)          # 過渡を捨てる
    lat40 = np.argmin(abs(th - np.radians(50)))
    per, npc = cycle_period(t, bu[:, lat40])
    out = dict(t_end=t[-1], period=per, ncyc=npc,
               DR=float(np.mean(h[late,4])),
               t90=float(np.abs(h[late,5]).max()),
               t60=float(np.abs(h[late,6]).max()),
               Bph=float(np.abs(h[late,1]).max()),
               Br=float(np.abs(h[late,2]).max()))
    ql = float(np.mean(h[late, 8+q.index('Q_Lambda')]))
    out['QL_Fsun'] = ql/3.846e33            # 太陽光度で規格化
    for name, key in KEYMAP[1:]:
        out[key] = float(np.mean(h[late, 8+q.index(name)]))/ql
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
            ('Q_eta / Q_Lam','Qeta','Qeta','%.3f')]
    print(f"  {'量':<28}{'計算':>10}{'論文':>10}   比")
    for label, rk, pk, fmt in rows:
        v = r.get(rk); pv = p.get(pk)
        if v is None: continue
        ratio = f"{v/pv:6.2f}" if pv else "     -"
        pvs = (fmt % pv) if pv is not None else "-"
        print(f"  {label:<28}{fmt % v:>10}{pvs:>10}   {ratio}")

if __name__ == '__main__':
    for spec in sys.argv[1:]:
        tag, a = spec.split(':')
        show(tag, float(a))
