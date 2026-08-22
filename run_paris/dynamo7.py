"""Rempel (2006) の非運動学的フラックス輸送ダイナモ.

流体を緩和させてから磁場を入れ、Lorentz 力のフィードバックまで含めて回す。
検証対象 (Rempel 2006 表1・図3・図4):
    周期 19 年 (運動学的) / 18 年 (非運動学的)
    max B_phi = 1.28 T (r=0.735 RSUN)
    max B_r   = 0.014 T (r=0.985 RSUN)
    トーショナル振動 4.7 nHz (極) / 3.5 nHz (緯度 60 度)
    蝶形図: 緯度 50 度から始まり 40 度でピーク
"""
import sys, os, time, numpy as np; sys.path.insert(0,'tests')
np.seterr(all='ignore')
import S2MFD
PARFILE=os.environ.get('S2MFD_PARFILE','parameters/rempel06.py')
from S2MFD.stratification import Stratification
from S2MFD.physics import conservative as cons, time_marching, poloidal_mag, boundary_condition
from S2MFD.physics.dynamic import DynamicSolver
from S2MFD.physics.energy import EnergyBudget, solar_luminosity
from conftest import make_cfg, make_grid

nx,ny   = int(sys.argv[1]), int(sys.argv[2])
relax_yr= float(sys.argv[3])     # 磁場を入れる前の流体緩和
dyn_yr  = float(sys.argv[4])     # ダイナモを回す年数
tag     = sys.argv[5]
kinematic = (len(sys.argv)>6 and sys.argv[6]=='kinematic')  # Lorentz 力を切る
over={}; init=None
for a in sys.argv[7:]:
    k,v=a.split('=')
    if k=='init': init=v
    else:
        # margin のように int でなければならない値があるので、
        # 小数点も指数もない文字列は int にする (1.0e12 等は float のまま)
        try: over[k]=int(v)
        except ValueError:
            try: over[k]=float(v)      # 1.0e12 のような指数表記も通す
            except ValueError: over[k]=v
bseed_arg=over.pop('bseed', 1.0)
om_bar_tau_yr=float(over.pop('om_bar_tau_yr', 18.0))  # 時間平均の時定数
outdir=f'results_rempel/{tag}'; os.makedirs(outdir, exist_ok=True)

# Rempel (2006) は運動学的参照解 (図 3) にだけ alpha クエンチングを使い、
# ローレンツ力を入れた解 (§3.1, 図 4, 表 1 列 3-9) では外している。
# コマンドラインで明示されていなければ、この規則に従う。
over.setdefault('alpha_quenching', bool(kinematic))
cfg=make_cfg(PARFILE,ix=nx,jx=ny, **over)
grid=make_grid(cfg); strat=Stratification(cfg,grid); setup=S2MFD.Setup(cfg,grid)
sol=DynamicSolver(cfg,grid,strat,setup); eb=EnergyBudget(cfg,grid,strat,setup)
m=grid.margin; sl=(slice(m,grid.ixg-m),slice(m,grid.jxg-m))
th=grid.th[m:grid.jxg-m]; rr=grid.rr[m:grid.ixg-m]
eq=np.argmin(abs(th-np.pi/2)); po=0
i735=np.argmin(abs(rr-0.735*cfg.RSUN)); isurf=len(rr)-1
lat60=np.argmin(abs(th-np.radians(30)))   # 緯度60度 = 余緯度30度
# hist の列: 0=t 1=|Bph|max 2=|Br|surf 3=Bph@0.735eq 4=DR
#   5=dOm(pole) 6=dOm(lat60) 7=E_B 8=Om1(pole)raw 9=Om1(lat60)raw
#   10 以降が QKEYS
QKEYS=['Q_Lambda','Q_nu_Omega','Q_C','Q_L_Omega','Q_nu_M','Q_B','Q_L_M','Q_eta']

print(f"[{tag}] alpha_quenching={getattr(cfg,'alpha_quenching',True)} "
      f"r_max={getattr(cfg,'r_max',grid.rrmax)/cfg.RSUN:.4f}R "
      f"thmax={cfg.thmax/np.pi:.3f}pi margin={grid.margin} "
      f"om_bar_tau={om_bar_tau_yr}yr", flush=True)
print(f"[{tag}] {nx}x{ny} alpha0={cfg.alpha0} kinematic={kinematic} "
      f"h={cfg.sld_fh} eps={cfg.sld_ep} cs_factor={cfg.sld_cs_factor}", flush=True)

# --- 第1段階: 流体の緩和 (磁場なし) ---------------------------------------
sol.magnetic=False
if init:
    dd=np.load(init)
    for k in ('om1','vrr','vth','ro1','se1'): getattr(sol,k)[:]=dd[k]
    print(f"[{tag}] 緩和済み状態 {init} から開始 (t={float(dd['t'])/3.156e7:.1f}yr 相当)", flush=True)
sol.set_primitive_from_conserved(sol.conserved())

# --- CFL 安全率の自動決定 -------------------------------------------------
# 明示されていなければ von Neumann の中立点 S0 の 0.9 倍を使う。
# 中央差分 + SSP-RK2 は拡散がないと無条件不安定なので、人工拡散や解像度を
# 変えたら安全率も変える必要がある (S2MFD/physics/stability.py 参照)。
# S0 は解像度とともに上がり (物理粘性の拡散数が効く)、人工拡散が弱い領域では
# cs_factor によらず一定になる (危険モードが 4-8 セルで SLD が効かない)。
if 'cfl_safety' not in over:
    _S0 = sol.neutral_cfl_safety()
    if _S0 == _S0:                      # nan でない
        cfg.cfl_safety = 0.9*_S0
        print(f"[{tag}] cfl_safety を自動設定: {cfg.cfl_safety:.3f} "
              f"(von Neumann 中立点 {_S0:.3f} の 0.9 倍)", flush=True)

dt=sol.cfl_dt()
ns=int(relax_yr*3.156e7/dt); t0=time.time(); t=0.0
for n in range(1,ns+1):
    sol.step(dt); t+=dt
    if n%2000==0: dt=min(dt, sol.cfl_dt())
    if n%(max(1,ns//8))==0:
        om=sol.om1[sl]
        print(f"[{tag}] 緩和 t={t/3.156e7:6.2f}yr DR={(om[-1,eq]-om[-1,po])/cfg.om0:+.4f} "
              f"vth={np.abs(sol.vth[sl]).max()/100:5.2f}m/s ({(time.time()-t0)/n*1e3:.2f}ms/st)", flush=True)
        if not np.all(np.isfinite(sol.om1)):
            print(f"[{tag}] *** 緩和中に発散 ***", flush=True); sys.exit(1)
np.savez(f'{outdir}/relaxed.npz', om1=sol.om1, vrr=sol.vrr, vth=sol.vth,
         ro1=sol.ro1, se1=sol.se1, t=t)

# 磁場の初期値。init に Bph/Aph が入っていればそこから**再開**する
# (final_state.npz を渡せば続きが回せる)。入っていなければ種磁場から始める。
_resume = None
if init:
    _d = np.load(init)
    if 'Bph' in _d and 'Aph' in _d:
        _resume = _d

# --- トーショナル振動の基準 Omega_bar ---------------------------------------
# Rempel (2006) 表 1 の量は max(Omega - Omega_bar) で、Omega_bar は**時間平均**
# (表 1 の注: "The maximum of Omega - Omega_bar and B_r is evaluated at
#  0.985 R_sun"、値は 12 サイクル平均で取られている)。
# 磁場を入れた瞬間の Omega で固定すると、ローレンツ力による差動回転の
# **永年的な減少**(表 1: 0.27 -> 0.21 -> 0.15 -> 0.10)がそのまま
# 「振動振幅」として計上され、一桁過大になる。
# ここでは指数移動平均で走る時間平均を作る。時定数は 1 サイクル (18 年)。
om_bar = (np.ascontiguousarray(_resume['om_bar'])
          if (_resume is not None and 'om_bar' in _resume) else sol.om1.copy())
om_bar_tau = om_bar_tau_yr*3.156e7

# --- 第2段階: 磁場を入れてダイナモ ----------------------------------------
sol.magnetic = not kinematic
legendre=S2MFD.Legendre(cfg,grid) if cfg.boundary_condition_type=='potential' else None
if _resume is not None:
    Bph = np.ascontiguousarray(_resume['Bph'])
    Aph = np.ascontiguousarray(_resume['Aph'])
    # 表示は物理セルだけで取る。ゴーストの**角**セルは境界条件のループが
    # 書かないので古い値が残っており (実測 9 T)、全体の max を出すと誤解する。
    # ステンシルは角を使わないので実害はない。
    print(f"[{tag}] 磁場も {init} から再開 "
          f"(max|Bph|={np.abs(Bph[sl]).max()*1e-4:.4f}T)", flush=True)
else:
    prof=np.exp(-((grid.RR-0.72*cfg.RSUN)/(0.05*cfg.RSUN))**2)
    bseed=float(bseed_arg)
    Bph=np.ascontiguousarray(bseed*prof*np.sin(2*grid.TH))  # 種磁場 [G]
    Aph=np.zeros_like(Bph)
sol.sync_to_induction()
pm=poloidal_mag(Aph,grid.RR,grid.sinTH,grid.drr,grid.dth)
sol.set_magnetic_field(pm[0],pm[1],Bph)
dt=sol.cfl_dt()
ns=int(dyn_yr*3.156e7/dt); nout=max(1,ns//2000)
hist=[]; butter=[]; t0=time.time()
for n in range(1,ns+1):
    Bph,Aph = time_marching(Bph,Aph,dt,cfg,grid,setup)
    # time_marching の出力ゴーストは未初期化 (np.empty_like) なので、
    # ゴーストを読む SLD フィルタの**前に**境界条件を掛ける。
    # 順序を逆にすると未初期化メモリを拾う (実測 相対 1e-10)。
    Bph,Aph = boundary_condition(Bph,Aph,cfg,grid,legendre)
    Bph,Aph = sol.magnetic_filter(Bph,Aph,dt)                 # Rempel 2014 のフィルタ段
    Bph,Aph = boundary_condition(Bph,Aph,cfg,grid,legendre)   # フィルタ後にもう一度
    pm=poloidal_mag(Aph,grid.RR,grid.sinTH,grid.drr,grid.dth)
    sol.set_magnetic_field(pm[0],pm[1],Bph)
    sol.step(dt); sol.sync_to_induction(); t+=dt
    if n%2000==0: dt=min(dt, sol.cfl_dt())
    # 走る時間平均 (指数移動平均). w = dt/tau で 1 次の低域通過.
    w_bar = min(1.0, dt/om_bar_tau)
    om_bar += w_bar*(sol.om1 - om_bar)
    if n%nout==0:
        if not np.all(np.isfinite(Bph)):
            print(f"[{tag}] *** ダイナモ中に発散 t={t/3.156e7:.2f}yr ***", flush=True); break
        B=Bph[sl]; Br=pm[0][sl]; om=sol.om1[sl]
        e=eb.reservoirs(sol.om1,sol.vrr,sol.vth,Bph)
        qq=eb.exchanges(sol.om1,sol.vrr,sol.vth,sol.brr,sol.bth,Bph,sol.se1)
        hist.append([t, np.abs(B).max()*1e-4, np.abs(Br[isurf]).max()*1e-4,
                     B[i735,eq], (om[-1,eq]-om[-1,po])/cfg.om0,
                     (om[-1,po]-om_bar[sl][-1,po])/(2*np.pi)*1e9,
                     (om[-1,lat60]-om_bar[sl][-1,lat60])/(2*np.pi)*1e9, e[2],
                     # 生の Omega_1 も残す。om_bar は時定数 1 サイクルの指数
                     # 移動平均なので、解がまだ緩和している間はドリフトに
                     # 追従しきれない。後処理で中心移動平均を引くために要る。
                     om[-1,po]/(2*np.pi)*1e9, om[-1,lat60]/(2*np.pi)*1e9]
                    +[qq[k] for k in QKEYS])
        butter.append(B[i735,:].copy())
        if n%(nout*100)==0:
            print(f"[{tag}] t={t/3.156e7:7.2f}yr |Bph|={hist[-1][1]:7.4f}T "
                  f"|Br|surf={hist[-1][2]:7.5f}T DR={hist[-1][4]:+.4f} "
                  f"dOm(pole)={hist[-1][5]:+6.2f}nHz ({(time.time()-t0)/n*1e3:.2f}ms/st)", flush=True)
        np.savez(f'{outdir}/dynamo.npz', hist=np.array(hist), butter=np.array(butter),
                 th=th, rr=rr, om0=cfg.om0, RSUN=cfg.RSUN, qkeys=np.array(QKEYS))
np.savez(f'{outdir}/final_state.npz', om1=sol.om1, vrr=sol.vrr, vth=sol.vth,
         ro1=sol.ro1, se1=sol.se1, Bph=Bph, Aph=Aph, om_bar=om_bar, t=t)
print(f"[{tag}] done {(time.time()-t0)/60:.1f} min", flush=True)
h=np.array(hist)
print(f"[{tag}] max|Bph| = {h[:,1].max():.4f} T (Rempel 図3: 1.28 T)", flush=True)
print(f"[{tag}] max|Br|surf = {h[:,2].max():.5f} T (Rempel: 0.014 T)", flush=True)
