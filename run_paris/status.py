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
    """(解像度, cs) ごとに DR をまとめる。

    以前はここで ``m(\d+)x(\d+)_cs(\d+)`` にしか当てず、後から投入した
    ``s216x144_*`` / ``s288x192_*`` を無視して**古い短いランの値**を
    表示していた (2026-08-23 に発見。216x144 cs=0.30 を「+0.2121@9y」と
    出していたが、実際は `s216x144_cs030` が 28 年走って +0.2777)。
    接頭辞によらず最長のランを採り、飽和判定も出す
    ``convergence.py`` に寄せた。
    """
    try:
        import convergence
    except ImportError:
        print("\n(convergence.py が見つからないので要約表を出さない)")
        return
    print()
    convergence.table(convergence.collect())


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else None)
    summary()
