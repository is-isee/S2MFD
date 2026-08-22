"""全ホストのランを 1 枚にまとめる。結果は共有 /scr/a000 にあるので
どのホストから実行しても同じものが見える。"""
import numpy as np, os, glob, re, sys
np.seterr(all='ignore'); YR = 3.156e7

def rows():
    for p in sorted(glob.glob('results_rempel/*/history.npz')):
        tag = p.split('/')[1]
        try: h = np.load(p)['hist']
        except Exception: continue
        if len(h) == 0: continue
        t = h[:, 0]/YR; vr = h[:, 3]
        yield dict(tag=tag, t=t[-1], DR=h[-1, 1], vr=vr[-1], vrmax=vr.max(),
                   vth=h[-1, 4], res=h[-1, 6], n=len(h),
                   bad=(abs(h[-1, 1]) > 0.6 or vr[-1] > 50))
    for p in sorted(glob.glob('results_rempel/*/dynamo.npz')):
        tag = p.split('/')[1]
        try: h = np.load(p)['hist']
        except Exception: continue
        if len(h) == 0: continue
        yield dict(tag=tag+' [dyn]', t=h[-1, 0]/YR, DR=h[-1, 4], vr=np.nan,
                   vrmax=np.abs(h[:, 1]).max(), vth=np.nan, res=np.nan,
                   n=len(h), bad=False)

def main(filt=None):
    print(f"{'tag':22}{'t[yr]':>8}{'DR':>10}{'vr':>7}{'vr最大':>8}{'残差':>10}  判定")
    for r in rows():
        if filt and filt not in r['tag']: continue
        res = '' if np.isnan(r['res']) else f"{r['res']:+.1e}"
        vr = '' if np.isnan(r['vr']) else f"{r['vr']:.2f}"
        print(f"{r['tag']:22}{r['t']:8.2f}{r['DR']:+10.4f}{vr:>7}{r['vrmax']:8.1f}"
              f"{res:>10}  {'*** 発散' if r['bad'] else 'OK'}")

def summary():
    """(解像度, cs) ごとに DR をまとめる。"""
    tab = {}
    for r in rows():
        m = re.match(r'm(\d+)x(\d+)_cs(\d+)$', r['tag'])
        if m:
            nx, ny, cs = int(m[1]), int(m[2]), float('0.'+m[3][1:]) if m[3][0]=='0' else float(m[3])/100
            tab[(nx, ny, cs)] = r
    if not tab: return
    print("\n=== 解像度 x 人工拡散 (DR @ 最新時刻) ===")
    css = sorted({k[2] for k in tab}); reso = sorted({(k[0], k[1]) for k in tab})
    print(f"{'格子':>10}" + "".join(f"{c:>16.2f}" for c in css))
    for nx, ny in reso:
        line = f"{nx}x{ny:<6}"
        for c in css:
            r = tab.get((nx, ny, c))
            line += f"{(f'{r[chr(68)+chr(82)]:+.4f}@{r[chr(116)]:.0f}y' if r else '-'):>16}"
        print(line)

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else None)
    summary()
