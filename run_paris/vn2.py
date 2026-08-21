"""SLD のスケール選択性を入れた von Neumann 解析。

SLD のフラックスは F = -0.5 c Phi_h (qR - qL) で、Phi_h は MC リミタの
再構成から決まる。滑らかな波では Phi_h -> 0 になるので、拡散は波数に
強く依存する。正弦波 u = sin(k x) に対して Phi_h(k) を数値的に測り、
実効拡散数 d(theta) = d_SLD * Phi_h(theta) * (qR-qL の効き) を入れる。
"""
import numpy as np

def sld_response(theta, ep=2.0, fh=2.0, n=2048):
    """波数 theta=k*dx の正弦波に SLD をかけたときの減衰率を返す。

    du_i/dt = -(F_{i+1} - F_i)/dx を計算し、u_i との比を取る
    (実効的な -kappa_eff k^2 に相当)。返すのは kappa_eff/(0.5*c*dx)。
    """
    i = np.arange(n)
    u = np.cos(theta*i)   # sin だと theta=pi で恒等的にゼロになる
    def mm3(a, b):
        c = 0.5*(a+b); A=ep*a; B=ep*b
        mx = np.maximum(np.maximum(A,B),c); mn = np.minimum(np.minimum(A,B),c)
        return np.where(mx<0, mx, 0.0) + np.where(mn>0, mn, 0.0)
    d0 = np.roll(u,-1)-u          # u[i+1]-u[i]
    dm = u-np.roll(u,1)           # u[i]-u[i-1]
    s_i  = mm3(dm, d0)            # セル i の傾き
    # 面 i (セル i-1 と i の境界)
    s_im = np.roll(s_i,1)
    ql = np.roll(u,1) + 0.5*s_im
    qr = u - 0.5*s_i
    dd = u - np.roll(u,1)
    dd = np.where(np.abs(dd)>1e-20, dd, 1e-20)
    ra = np.clip((qr-ql)/dd, None, 1.0)
    pp = np.where(ra<=0.0, 0.0, np.maximum(0.0, 1.0+fh*(ra-1.0)))
    F = -0.5*pp*(qr-ql)           # c=1, jac=1 の単位
    dudt = -(np.roll(F,-1)-F)     # /dx は外で
    # kappa_eff k^2 u = -dudt/dx  ->  kappa_eff/(0.5 c dx) を返す
    k = 40  # 中央付近のセルで評価
    with np.errstate(divide='ignore', invalid='ignore'):
        val = -dudt/np.where(np.abs(u)>1e-12, u, np.nan)
    v = np.nanmean(val[100:n-100])
    # 4 sin^2(theta/2) で割れば「拡散係数比」になる
    return v/(4*np.sin(0.5*theta)**2)/0.5 if theta>1e-8 else 0.0

th=np.linspace(1e-6,np.pi,2001)
resp=np.array([sld_response(t) for t in th])
resp=np.clip(np.nan_to_num(resp),0,None)
print("SLD の実効拡散 (格子スケールでの値を 1 とした比)")
print(f"{'波長[セル]':>11}{'theta':>8}{'Phi 実効':>11}")
for L in (2,2.5,3,4,6,8,12,20,40):
    t=2*np.pi/L; i=np.argmin(abs(th-t))
    print(f"{L:11.1f}{t:8.3f}{resp[i]:11.4f}")

def gmax(nu, dsld):
    d = dsld*np.interp(th, th, resp)
    z = -4.0*d*np.sin(0.5*th)**2 - 1j*nu*np.sin(th)
    return np.abs(1.0+z+0.5*z*z).max()

print("\nスケール選択性を入れた安定限界 (d_SLD = 0.15*S)")
print(f"{'S':>6}{'d_SLD':>9}{'|G|max':>11}{'判定':>8}")
for S in (0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.0):
    g=gmax(S, 0.15*S)
    print(f"{S:6.2f}{0.15*S:9.4f}{g:11.6f}{'  安定' if g<=1.0 else '  不安定':>8}")
lo,hi=0.1,2.0
for _ in range(50):
    mid=0.5*(lo+hi)
    if gmax(mid,0.15*mid)<=1.0: lo=mid
    else: hi=mid
print(f"\n理論的な安定限界 S = {lo:.3f}   (実測: 0.6 は安定、0.8 は発散)")
