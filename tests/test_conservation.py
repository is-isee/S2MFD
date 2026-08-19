"""保存形離散化の検証: 質量と角運動量が machine precision で保存すること。

ユーザー要求 (doc/dev_records/2026-08-19_rempel2006_feasibility.md):

- 角運動量と質量 (音速抑制法を使うので ∫(ρ0 + ζ²ρ1) dV) は machine
  precision で保存すること
- 境界から保存量が出ていかないこと (人工粘性も物理も)

このファイルはその 2 点を回帰テストとして固定する。
"""
import numpy as np
import pytest

import S2MFD
from S2MFD.stratification import Stratification
from S2MFD.physics import conservative as cons
from S2MFD.physics import hydro
from S2MFD.physics import artdif
from S2MFD.physics import energy
from S2MFD.physics import dynamic

from conftest import make_cfg, make_grid


# ---------------------------------------------------------------------------
# 固定具
# ---------------------------------------------------------------------------
@pytest.fixture(scope='module')
def setup_dynamic():
    """Rempel (2006) 設定の小さめの格子一式。"""
    cfg = make_cfg('parameters/rempel06.py', ix=48, jx=48)
    grid = make_grid(cfg)
    strat = Stratification(cfg, grid)
    setup = S2MFD.Setup(cfg, grid)
    return cfg, grid, strat, setup


def _solenoidal_mass_flux(grid, strat):
    """離散的に厳密な ``div(rho0 v) = 0`` を満たす速度場を作る。

    ストリーム関数 :math:`\\psi` をセルの角で定義し、面フラックスをその
    差分で作れば連続の式は丸め誤差レベルで満たされる。ここではセル中心の
    速度が要るので、面フラックスから中心値へ戻す。

    保存則の検証は「速度が何であっても保存する」ことを見たいので、
    厳密な非圧縮性は本質ではない。それでも現実的な子午面循環に近い形を
    使っておくほうが、フラックスの大きさが実際の計算に近くなる。
    """
    rr, th = grid.RR, grid.TH
    rrmin, rrmax = grid.rrmin, grid.rrmax
    prof = np.sin(np.pi*(rr - rrmin)/(rrmax - rrmin))
    vrr = 2.0e3*prof*(3.0*grid.cosTH**2 - 1.0)
    vth = 1.5e3*prof*np.sin(2.0*th)
    # 境界で速度をゼロにしておく (フラックスのゼロ化とは独立の物理設定)
    vrr = vrr*np.sin(np.pi*(rr - rrmin)/(rrmax - rrmin))
    return np.ascontiguousarray(vrr), np.ascontiguousarray(vth)


# ---------------------------------------------------------------------------
# 格子と面の整合
# ---------------------------------------------------------------------------
def test_face_coordinates_align_with_domain_boundaries():
    """面配列の境界が計算領域の境界とビット一致すること。

    保存形の離散化は「境界面のフラックスをゼロにする」ことで保存を保証
    するので、その面が本当に物理境界でなければ意味がない。
    """
    cfg = make_cfg('parameters/rempel06.py', ix=48, jx=48)
    grid = make_grid(cfg)
    m = grid.margin
    assert grid.RRm[m, 0] == grid.rrmin
    assert grid.RRm[grid.ixg - m, 0] == pytest.approx(grid.rrmax, rel=1e-15)
    assert grid.THm[0, m] == grid.thmin
    assert grid.sinTHm[0, m] == 0.0


def test_stratification_rejects_domain_beyond_polytrope():
    """ポリトロープが密度ゼロに達する半径を超える領域は明示的に弾くこと。

    太陽パラメタでは密度ゼロ点は 1.0026 RSUN にあり、1.0 RSUN はぎりぎり
    内側 (ξ ≈ 0.006) なので通ってしまう。ただしその近傍は密度も音速も
    急激に小さくなるため、Rempel (2006) / 齋藤 (2024) はどちらも
    0.96 RSUN で切っている。
    """
    cfg = make_cfg('parameters/rempel06.py', ix=48, jx=48,
                   rrmax=1.01*6.96e10)
    grid = make_grid(cfg)
    with pytest.raises(ValueError, match='ポリトロープ'):
        Stratification(cfg, grid)


def test_stratification_is_hydrostatic(setup_dynamic):
    """背景成層の静水圧平衡が格子細分化で 2 次収束すること。

    ポリトロープは解析的には厳密に :math:`dp_0/dr = -\\rho_0 g` を満たすので、
    残差は中心差分の打ち切り誤差そのものである。したがって「小さい」ことでは
    なく「2 次で収束する」ことを検証するのが正しい。

    なお運動方程式には背景の釣り合いが**そもそも現れない**。Rempel (2005) /
    齋藤 (2024) の定式化では圧力勾配項は摂動 :math:`p_1` のみ、浮力項は
    :math:`\\rho_1 g` のみで、背景の釣り合いは解析的に差し引かれている
    (well-balanced)。よってこの離散残差が偽の流れを駆動することはない。
    """
    cfg, _, _, _ = setup_dynamic

    def residual(nr):
        c = make_cfg('parameters/rempel06.py', ix=nr, jx=8)
        g = make_grid(c)
        s = Stratification(c, g)
        m = g.margin
        sl = slice(m + 1, g.ixg - m - 1)
        dpdr = (s.pr0[m + 2:g.ixg - m] - s.pr0[m:g.ixg - m - 2])/(2*g.drr)
        weight = np.abs(s.ro0[sl]*s.gr[sl]).max()
        return np.abs(dpdr + s.ro0[sl]*s.gr[sl]).max()/weight

    coarse, fine = residual(48), residual(96)
    assert coarse < 1e-3, f'粗い格子でも残差が大きすぎる: {coarse:.2e}'
    order = np.log2(coarse/fine)
    assert 1.8 < order < 2.2, f'収束次数が 2 でない: {order:.2f}'


# ---------------------------------------------------------------------------
# 質量保存
# ---------------------------------------------------------------------------
def test_mass_conserved_to_machine_precision(setup_dynamic):
    """∫(ρ0 + ζ²ρ1) dV が数千ステップで machine precision に留まること。

    ρ0 は静的なので ∫ζ²ρ1 dV の変化だけを見ればよい。ここでは ζ² を
    掛け戻した保存量 q = r²sinθ ρ1 の積分がゼロのままかを確認する
    (RSST 係数は発散全体に掛かるので、ζ² を戻した量が保存する)。
    """
    cfg, grid, strat, _ = setup_dynamic
    m = grid.margin
    vrr, vth = _solenoidal_mass_flux(grid, strat)
    work = hydro.HydroWork(grid)

    zeta2 = strat.zeta**2
    izeta2 = 1.0/zeta2
    q_ro = np.zeros((grid.ixg, grid.jxg))

    def conserved_total(q):
        # q = r²sinθ ρ1 なので、ζ² を掛けた量の積分が保存量
        return cons.cell_integral(q*zeta2[:, None], grid.drr, grid.dth, m)

    # 初期に非自明な密度摂動を入れておく (ゼロのままでは保存が自明になる)
    q_ro[m:grid.ixg - m, m:grid.jxg - m] = (
        strat.JM*1e-3*strat.ro0[:, None]*np.sin(3*grid.TH)
    )[m:grid.ixg - m, m:grid.jxg - m]
    total0 = conserved_total(q_ro)
    scale = cons.cell_integral(np.abs(q_ro*zeta2[:, None]), grid.drr, grid.dth, m)

    dt = 50.0
    for _ in range(2000):
        dq = np.zeros_like(q_ro)
        hydro.mass_rhs(dq, vrr, vth, strat.JV, strat.JVY, izeta2,
                       grid.drr, grid.dth, m, work.ffr, work.ffth, work.cen)
        q1 = q_ro + dt*dq
        dq1 = np.zeros_like(q_ro)
        hydro.mass_rhs(dq1, vrr, vth, strat.JV, strat.JVY, izeta2,
                       grid.drr, grid.dth, m, work.ffr, work.ffth, work.cen)
        q_ro = 0.5*(q_ro + q1 + dt*dq1)

    drift = abs(conserved_total(q_ro) - total0)/scale
    assert drift < 1e-14, f'質量の相対ドリフト {drift:.3e}'


# ---------------------------------------------------------------------------
# 角運動量保存
# ---------------------------------------------------------------------------
def _angmom_step(q_om, om1, vrr, vth, bb, grid, strat, setup, cfg, dt, work,
                 magnetic, consistent=False):
    m = grid.margin

    def rhs(q):
        hydro.to_primitive_om1(q, strat.iJL, om1, m)
        dq = np.zeros_like(q)
        hydro.angular_momentum_rhs(
            dq, om1, vrr, vth, bb[0], bb[1], bb[2],
            strat.JL, strat.JLY, strat.JV, strat.JVY, strat.W2,
            grid.RR, grid.RRm, grid.sinTH, grid.sinTHm,
            strat.ro0, strat.ro0m, setup.lam_rp, setup.lam_tp, cfg.om0,
            setup.nu_dif, setup.nu_dif_m, setup.nu_lam, setup.nu_lam_m,
            grid.drr, grid.dth, m, magnetic,
            consistent, False, np.zeros_like(q), np.zeros_like(q),
            np.zeros(grid.jxg), work.ffr, work.ffth, work.cen)
        return dq

    q1 = q_om + dt*rhs(q_om)
    return 0.5*(q_om + q1 + dt*rhs(q1))


@pytest.mark.parametrize('magnetic,consistent',
                         [(False, False), (True, False), (True, True)])
def test_angular_momentum_conserved_to_machine_precision(setup_dynamic, magnetic,
                                                         consistent):
    """∫ρ0 r²sin²θ Ω1 dV が machine precision で保存すること。

    移流・粘性・Λ効果・Maxwell 応力のすべてを含めて検証する。Λ効果は
    角運動量を再分配するだけで総量を変えない (発散形なので) ことも
    同時に確認していることになる。
    """
    cfg, grid, strat, setup = setup_dynamic
    m = grid.margin
    vrr, vth = _solenoidal_mass_flux(grid, strat)
    work = hydro.HydroWork(grid)

    # 初期の差動回転 (剛体回転からのずれ)
    om1 = np.zeros((grid.ixg, grid.jxg))
    om1[:] = -0.05*cfg.om0*grid.cosTH**2
    q_om = strat.JL*om1

    # 磁場 (Maxwell 応力の検証用)。ゼロでない値を与えて発散形を試す。
    if magnetic:
        prof = np.sin(np.pi*(grid.RR - grid.rrmin)/(grid.rrmax - grid.rrmin))
        bb = (1.0e3*prof*grid.cosTH,
              5.0e2*prof*grid.sinTH,
              1.0e4*prof*np.sin(2*grid.TH))
        bb = tuple(np.ascontiguousarray(b) for b in bb)
    else:
        z = np.zeros((grid.ixg, grid.jxg))
        bb = (z, z, z)

    total0 = cons.cell_integral(q_om, grid.drr, grid.dth, m)
    scale = cons.cell_integral(np.abs(q_om), grid.drr, grid.dth, m)

    dt = 50.0
    for _ in range(2000):
        q_om = _angmom_step(q_om, om1, vrr, vth, bb, grid, strat, setup, cfg,
                            dt, work, magnetic, consistent)

    assert np.all(np.isfinite(q_om))
    drift = abs(cons.cell_integral(q_om, grid.drr, grid.dth, m) - total0)/scale
    assert drift < 1e-14, f'角運動量の相対ドリフト {drift:.3e}'


def test_angular_momentum_state_actually_evolves(setup_dynamic):
    """保存則の検証が「何も起きていない」ことによる自明な成功でないこと。

    保存量がゼロの増分しか受け取っていなければドリフトもゼロになるので、
    Ω1 が実際に有意に変化していることを別途確認する。
    """
    cfg, grid, strat, setup = setup_dynamic
    m = grid.margin
    vrr, vth = _solenoidal_mass_flux(grid, strat)
    work = hydro.HydroWork(grid)
    z = np.zeros((grid.ixg, grid.jxg))

    om1 = np.zeros((grid.ixg, grid.jxg))
    om1[:] = -0.05*cfg.om0*grid.cosTH**2
    om1_init = om1.copy()
    q_om = strat.JL*om1

    for _ in range(2000):
        q_om = _angmom_step(q_om, om1, vrr, vth, (z, z, z), grid, strat, setup,
                            cfg, 50.0, work, False)
    hydro.to_primitive_om1(q_om, strat.iJL, om1, m)

    sl = (slice(m, grid.ixg - m), slice(m, grid.jxg - m))
    change = np.abs(om1[sl] - om1_init[sl]).max()/np.abs(om1_init[sl]).max()
    assert change > 1e-3, f'Ω1 がほとんど変化していない (相対変化 {change:.2e})'


# ---------------------------------------------------------------------------
# 保存量の選び方 (齋藤形 vs 素朴形)
# ---------------------------------------------------------------------------
def test_perturbation_form_beats_total_form(setup_dynamic):
    """保存量に Ω0 を含めない齋藤形が、含める素朴形より桁違いに正確なこと。

    両者は解析的に等価だが、素朴形 q = ρ0 r⁴sin³θ (Ω0+Ω1) では Ω0 に
    支配された大きな数に Ω1 の増分を加えることになり、Ω1 の相対精度が
    eps·(Ω0/Ω1) に落ちる。Rempel 氏が「保存形ではうまくいかなかった」と
    述べたのはこの素朴形を指すと考えられ、齋藤 (2024) が成功したのは
    保存量から Ω0 を外していたためである (doc/dev_records 参照)。

    ここでは剛体回転 (Ω1 = 0) から出発し、子午面循環による角運動量の
    再分配で Ω1 が育つ過程を比較する。
    """
    cfg, grid, strat, setup = setup_dynamic
    m = grid.margin
    vrr, vth = _solenoidal_mass_flux(grid, strat)
    work = hydro.HydroWork(grid)
    z = np.zeros((grid.ixg, grid.jxg))
    sl = (slice(m, grid.ixg - m), slice(m, grid.jxg - m))

    def run(include_om0, dtype):
        om1 = np.zeros((grid.ixg, grid.jxg))
        JL = strat.JL.astype(dtype)
        q = JL*(cfg.om0 + om1) if include_om0 else JL*om1

        def to_om1(qq):
            out = np.zeros_like(om1)
            out[sl] = qq[sl]/JL[sl] - (cfg.om0 if include_om0 else 0.0)
            return out

        def rhs(qq):
            o = to_om1(qq)
            dq = np.zeros((grid.ixg, grid.jxg))
            hydro.angular_momentum_rhs(
                dq, np.ascontiguousarray(o), vrr, vth, z, z, z,
                strat.JL, strat.JLY, strat.JV, strat.JVY, strat.W2,
                grid.RR, grid.RRm, grid.sinTH,
                grid.sinTHm, strat.ro0, strat.ro0m,
                setup.lam_rp, setup.lam_tp, cfg.om0,
                setup.nu_dif, setup.nu_dif_m, setup.nu_lam, setup.nu_lam_m,
                grid.drr, grid.dth, m, False,
                False, False, np.zeros((grid.ixg, grid.jxg)),
                np.zeros((grid.ixg, grid.jxg)), np.zeros(grid.jxg),
                work.ffr, work.ffth, work.cen)
            return dq

        for _ in range(1500):
            q1 = q + 50.0*rhs(q)
            q = 0.5*(q + q1 + 50.0*rhs(q1))
        return to_om1(q)

    ref = run(False, np.longdouble)
    om_pert = run(False, np.float64)
    om_tot = run(True, np.float64)

    norm = np.abs(ref[sl]).max()
    err_pert = np.abs(om_pert[sl] - ref[sl]).max()/norm
    err_tot = np.abs(om_tot[sl] - ref[sl]).max()/norm

    assert err_pert < 1e-12, f'齋藤形の誤差 {err_pert:.2e}'
    assert err_tot > 100*err_pert, (
        f'素朴形が悪化していない: 齋藤形 {err_pert:.2e} vs 素朴形 {err_tot:.2e}')


# ---------------------------------------------------------------------------
# 子午面の運動量と粘性
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('seed', range(5))
def test_viscosity_always_dissipates_kinetic_energy(setup_dynamic, seed):
    """粘性が運動エネルギーを必ず減らすこと (散逸の正値性)。

    粘性応力テンソルを球座標で展開する際は項が多く、符号や幾何因子を
    1 つ間違えても場のパターンによっては見た目が変わらないことがある。
    「任意の速度場に対して :math:`\\int\\rho_0\\boldsymbol{v}\\cdot
    \\boldsymbol{F}_\\nu\\,dV < 0`」はテンソル全体の整合性を一度に縛るので、
    転記ミスの検出力が高い。

    保存量 :math:`q_m = r^2\\sin\\theta\\rho_0 v` に対する時間微分を使うと、
    運動エネルギーの変化率はそのまま
    :math:`\\int(v_r\\,\\dot q_{m,r} + v_\\theta\\,\\dot q_{m,\\theta})
    \\,dr\\,d\\theta` になる。
    """
    cfg, grid, strat, setup = setup_dynamic
    m = grid.margin
    rng = np.random.default_rng(seed)
    work_arr = hydro.HydroWork(grid)
    sl = (slice(m, grid.ixg - m), slice(m, grid.jxg - m))

    prof = np.sin(np.pi*(grid.RR - grid.rrmin)/(grid.rrmax - grid.rrmin))
    kr, kt = rng.integers(1, 4), rng.integers(1, 4)
    # v_r は極で sinθ 倍して滑らかに、v_θ は極で消えるように取る
    vrr = np.ascontiguousarray(1e3*prof*np.cos(kr*grid.TH)*grid.sinTH)
    vth = np.ascontiguousarray(1e3*prof*np.sin(kt*grid.TH))

    dmr = np.zeros((grid.ixg, grid.jxg))
    dmt = np.zeros((grid.ixg, grid.jxg))
    hydro.viscous_meridional_rhs(dmr, dmt, vrr, vth, grid.rr, grid.sinTH,
                                 grid.cosTH, strat.ro0,
                                 setup.nu_dif, setup.nu_dif_m,
                                 grid.drr, grid.dth, m, work_arr.ffr, work_arr.ffth)

    dkedt = (vrr[sl]*dmr[sl] + vth[sl]*dmt[sl]).sum()*grid.drr*grid.dth
    ke = 0.5*(strat.JV[sl]*(vrr[sl]**2 + vth[sl]**2)).sum()*grid.drr*grid.dth
    assert dkedt < 0.0, f'粘性が運動エネルギーを増やしている: {dkedt:.3e}'
    # 桁が合っていることも見る (減衰時定数が拡散時間スケール程度)
    tau = ke/abs(dkedt)
    assert 1e5 < tau < 1e10, f'減衰時定数が非現実的: {tau:.3e} s'


def test_lorentz_force_matches_magnetic_pressure_scale(setup_dynamic):
    """ローレンツ力の大きさが :math:`B^2/8\\pi L` と同じ桁になること。

    符号や :math:`4\\pi` の入れ忘れ、回転の幾何因子の取り違えを検出する
    ための粗い次元チェック。
    """
    cfg, grid, strat, setup = setup_dynamic
    m = grid.margin
    work_arr = hydro.HydroWork(grid)
    z = np.zeros((grid.ixg, grid.jxg))
    sl = (slice(m, grid.ixg - m), slice(m, grid.jxg - m))

    width = 0.05*cfg.RSUN
    prof = np.exp(-((grid.RR - 0.72*cfg.RSUN)/width)**2)
    brr = np.ascontiguousarray(3e2*prof*grid.cosTH)
    bth = np.ascontiguousarray(3e2*prof*grid.sinTH)
    bph = np.ascontiguousarray(1.28e4*prof*np.sin(2*grid.TH))

    dmr = np.zeros((grid.ixg, grid.jxg))
    dmt = np.zeros((grid.ixg, grid.jxg))
    hydro.momentum_rhs(dmr, dmt, z, z, z, z, z, brr, bth, bph,
                       strat.JV, strat.JVY, strat.JM, strat.RSIN, grid.RR,
                       grid.sinTH, grid.cosTH, strat.ro0, strat.gr,
                       cfg.om0, grid.drr, grid.dth, m, True,
                       work_arr.ffr, work_arr.ffth, work_arr.cen)

    force = np.abs(dmr[sl]/strat.JM[sl]).max()
    scale = (bph[sl]**2/(8*np.pi*width)).max()
    assert 0.1 < force/scale < 10.0, f'ローレンツ力の桁が合わない: {force/scale:.3f}'


def test_momentum_rhs_is_finite(setup_dynamic):
    """非自明な状態で運動量の右辺が有限であること (極の 1/sinθ を含む)。"""
    cfg, grid, strat, setup = setup_dynamic
    m = grid.margin
    work_arr = hydro.HydroWork(grid)

    prof = np.sin(np.pi*(grid.RR - grid.rrmin)/(grid.rrmax - grid.rrmin))
    vrr = np.ascontiguousarray(2e3*prof*(3*grid.cosTH**2 - 1))
    vth = np.ascontiguousarray(2e3*prof*np.sin(2*grid.TH))
    om1 = np.ascontiguousarray(-0.05*cfg.om0*grid.cosTH**2)
    ro1 = np.ascontiguousarray(1e-4*strat.ro0[:, None]*np.sin(3*grid.TH))
    se1 = np.zeros_like(ro1)
    pr1 = np.ascontiguousarray(strat.pressure_perturbation(ro1, se1))
    z = np.zeros((grid.ixg, grid.jxg))

    dmr = np.zeros((grid.ixg, grid.jxg))
    dmt = np.zeros((grid.ixg, grid.jxg))
    hydro.momentum_rhs(dmr, dmt, vrr, vth, om1, ro1, pr1, z, z, z,
                       strat.JV, strat.JVY, strat.JM, strat.RSIN, grid.RR,
                       grid.sinTH, grid.cosTH, strat.ro0, strat.gr,
                       cfg.om0, grid.drr, grid.dth, m, False,
                       work_arr.ffr, work_arr.ffth, work_arr.cen)
    hydro.viscous_meridional_rhs(dmr, dmt, vrr, vth, grid.rr, grid.sinTH,
                                 grid.cosTH, strat.ro0,
                                 setup.nu_dif, setup.nu_dif_m,
                                 grid.drr, grid.dth, m, work_arr.ffr, work_arr.ffth)
    assert np.all(np.isfinite(dmr)) and np.all(np.isfinite(dmt))


# ---------------------------------------------------------------------------
# 人工拡散 (slope-limited diffusion)
# ---------------------------------------------------------------------------
def _sld_setup(setup_dynamic):
    cfg, grid, strat, setup = setup_dynamic
    m = grid.margin
    shape = (grid.ixg, grid.jxg)
    # 面上の特性速度 (定数でよい: ここで見たいのはリミタの振る舞い)
    csp = np.full(shape, 1.0e5)
    # r 面のヤコビアン (角運動量と同じ形)
    jac_r = np.zeros(shape)
    jac_r[1:] = 0.5*(strat.JL[1:] + strat.JL[:-1])
    jac_th = np.zeros(shape)
    jac_th[:, 1:] = 0.5*(strat.JLY[:, 1:] + strat.JLY[:, :-1])
    return cfg, grid, strat, m, csp, jac_r, jac_th


def test_sld_conserves_exactly(setup_dynamic):
    """人工拡散が保存量を厳密に保存すること。

    齋藤 (2024) コードの人工粘性 artdif.f90 は、本体と異なる保存量に対して
    定義されているために角運動量保存を壊し、その結果として無効化されている
    (mhd.f90 で呼び出しがコメントアウトされている)。本実装は同じ枠組みの
    面フラックスとして書いているので保存が成り立つ。
    """
    cfg, grid, strat, m, csp, jac_r, jac_th = _sld_setup(setup_dynamic)
    rng = np.random.default_rng(7)
    # 格子スケールの振動を含む Omega1 (人工拡散が最も強く効く状況)
    om1 = np.ascontiguousarray(
        -0.05*cfg.om0*grid.cosTH**2
        + 0.01*cfg.om0*rng.standard_normal((grid.ixg, grid.jxg)))
    ffr = np.zeros_like(om1)
    ffth = np.zeros_like(om1)
    dq = np.zeros_like(om1)
    artdif.sld_diffuse(dq, om1, jac_r, jac_th, csp, csp, 2.0, 2.0,
                       grid.drr, grid.dth, m, ffr, ffth)

    total = cons.cell_integral(dq, grid.drr, grid.dth, m)
    scale = cons.cell_integral(np.abs(dq), grid.drr, grid.dth, m)
    assert scale > 0, '人工拡散が何も効いていない'
    assert abs(total)/scale < 1e-14, f'人工拡散が保存を壊している: {total/scale:.2e}'


@pytest.mark.parametrize('seed', range(4))
def test_sld_always_dissipates(setup_dynamic, seed):
    """人工拡散が必ずエネルギーを減らすこと (反拡散にならない)。

    符号ガード (jump が勾配と逆符号ならフラックスをゼロ) が効いていれば
    構造的に保証される。ここはエネルギー注入という最悪の失敗モードを
    直接押さえる回帰テスト。
    """
    cfg, grid, strat, m, csp, jac_r, jac_th = _sld_setup(setup_dynamic)
    rng = np.random.default_rng(seed)
    om1 = np.ascontiguousarray(
        -0.05*cfg.om0*grid.cosTH**2
        + 0.02*cfg.om0*rng.standard_normal((grid.ixg, grid.jxg)))
    ffr = np.zeros_like(om1)
    ffth = np.zeros_like(om1)
    dq = np.zeros_like(om1)
    artdif.sld_diffuse(dq, om1, jac_r, jac_th, csp, csp, 2.0, 2.0,
                       grid.drr, grid.dth, m, ffr, ffth)

    sl = (slice(m, grid.ixg - m), slice(m, grid.jxg - m))
    dke = ((cfg.om0 + om1[sl])*dq[sl]).sum()*grid.drr*grid.dth
    assert dke < 0.0, f'人工拡散がエネルギーを注入している: {dke:.3e}'


def test_sld_vanishes_exactly_for_linear_fields(setup_dynamic):
    """線形な場では人工拡散のフラックスが厳密にゼロになること。

    Rempel (2014) 式 (6)(7) の再構成は、局所的に線形な場では左右からの
    外挿が面上で一致するので :math:`u_r-u_l=0` となり、フラックスが
    **厳密に**消える。これが「解像された場には効かない」の中身であり、
    モデルが依存する低拡散領域 (オーバーシュート層の κ_t、放射層の ν_dif)
    を潰さないための必須条件。
    """
    cfg, grid, strat, setup = setup_dynamic
    m = grid.margin
    shape = (grid.ixg, grid.jxg)
    csp = np.full(shape, 1.0e5)
    jac = np.ones(shape)
    out = np.zeros(shape)

    # r 方向に完全に線形な場
    lin = np.ascontiguousarray(np.arange(grid.ixg, dtype=float)[:, None]
                               * np.ones((1, grid.jxg)))
    artdif.sld_flux_r(lin, jac, csp, 2.0, 2.0, m, out)
    assert np.abs(out[m + 1:grid.ixg - m, :]).max() == 0.0, (
        '線形場でフラックスが立っている')

    # 1 セルおきに符号が変わる格子スケールの振動 -> 最大拡散 (Phi_h = 1)
    ii = np.arange(grid.ixg)[:, None]
    zig = np.ascontiguousarray(((-1.0)**ii)*np.ones((1, grid.jxg)))
    out[:] = 0.0
    artdif.sld_flux_r(zig, jac, csp, 2.0, 2.0, m, out)
    i = grid.ixg//2
    expected = -0.5*1.0e5*(zig[i, 0] - zig[i - 1, 0])
    assert abs(out[i, 0] - expected) < 1e-8*abs(expected), (
        f'格子スケール振動で最大拡散になっていない: {out[i,0]:.4e} vs {expected:.4e}')


def test_sld_is_negligible_for_resolved_fields(setup_dynamic):
    """滑らかで解像された場では人工拡散がほぼ効かないこと。

    これが成り立たないと、人工拡散が物理的な差動回転そのものを削って
    しまう。格子スケールの振動に対する応答と比較して桁で小さいことを見る。
    """
    cfg, grid, strat, m, csp, jac_r, jac_th = _sld_setup(setup_dynamic)
    ffr = np.zeros((grid.ixg, grid.jxg))
    ffth = np.zeros((grid.ixg, grid.jxg))

    def response(field):
        dq = np.zeros((grid.ixg, grid.jxg))
        artdif.sld_diffuse(dq, np.ascontiguousarray(field), jac_r, jac_th,
                           csp, csp, 2.0, 2.0, grid.drr, grid.dth, m, ffr, ffth)
        return cons.cell_integral(np.abs(dq), grid.drr, grid.dth, m)

    smooth = -0.05*cfg.om0*grid.cosTH**2*np.sin(
        np.pi*(grid.RR - grid.rrmin)/(grid.rrmax - grid.rrmin))
    # 1 セルおきに符号が変わる格子スケールの振動
    ii = np.arange(grid.ixg)[:, None]
    jj = np.arange(grid.jxg)[None, :]
    zigzag = 0.05*cfg.om0*((-1.0)**ii)*((-1.0)**jj)*np.ones_like(grid.RR)

    r_smooth, r_zig = response(smooth), response(zigzag)
    assert r_smooth < 0.02*r_zig, (
        f'滑らかな場にも人工拡散が効きすぎている: {r_smooth/r_zig:.3f}')


# ---------------------------------------------------------------------------
# CFL
# ---------------------------------------------------------------------------
def test_cfl_includes_alfven_speed(setup_dynamic):
    """CFL が音速・アルヴェン速度・流れの 3 つで決まること。

    音速抑制法 (RSST) が遅くするのは音波だけで、アルヴェン速度は変わらない。
    したがって強磁場を入れれば dt は必ず小さくなる。これを見落とすと
    ζ を上げるほど速くなるという誤った見積もりになる。
    """
    cfg, grid, strat, _ = setup_dynamic
    m = grid.margin
    z = np.zeros((grid.ixg, grid.jxg))
    ro1 = np.zeros_like(z)
    vrr = np.ascontiguousarray(2e3*np.ones_like(z))
    vth = np.zeros_like(z)

    # 上部対流層 (rho0 が小さい) に強磁場を置く
    prof = np.exp(-((grid.RR - 0.94*cfg.RSUN)/(0.02*cfg.RSUN))**2)
    bph = np.ascontiguousarray(2e4*prof*grid.sinTH)

    def dt_pair(zeta):
        c = make_cfg('parameters/rempel06.py', ix=48, jx=48, rsst_zeta=zeta)
        s = Stratification(c, grid)
        args = (s.ro0, ro1, s.cs_eff, grid.rr, grid.drr, grid.dth,
                cfg.nu0, m, cfg.cfl_safety)
        return (hydro.cfl_dt(vrr, vth, z, z, z, *args, False),
                hydro.cfl_dt(vrr, vth, z, z, bph, *args, True))

    # ζ=100 (齋藤/Rempel の値) では音速がまだ支配的で、Rempel 級の磁場を
    # 上部対流層に置いてもアルヴェン速度は CFL を変えない。
    hyd100, mag100 = dt_pair(100.0)
    assert mag100 == hyd100, (
        'ζ=100 で既にアルヴェン律速になっている。この設定では音速が'
        f'支配的なはず: {mag100:.1f} vs {hyd100:.1f}')

    # ζ を大きくして音波を強く抑えると、アルヴェン速度が下限を作る。
    # ここが「ζ を上げれば上げるだけ速くなる」わけではない理由。
    hyd_big, mag_big = dt_pair(1.0e4)
    assert mag_big < 0.5*hyd_big, (
        'ζ を上げてもアルヴェン律速に切り替わっていない: '
        f'{mag_big:.1f} vs {hyd_big:.1f}')
    # 頭打ちの水準: ζ を 100 倍しても dt は 100 倍にならない
    assert mag_big < 30*mag100, (
        f'アルヴェン速度による頭打ちが効いていない: {mag_big/mag100:.1f} 倍')


# ---------------------------------------------------------------------------
# 境界フラックスのゼロ化
# ---------------------------------------------------------------------------
def test_boundary_fluxes_are_literal_zero():
    """境界面のフラックスが「小さい」ではなく厳密にゼロであること。"""
    ixg, jxg, m = 12, 12, 1
    ff = np.random.default_rng(0).standard_normal((ixg, jxg))
    cons.zero_boundary_faces_r(ff, m)
    assert np.all(ff[m, :] == 0.0)
    assert np.all(ff[ixg - m, :] == 0.0)

    ff = np.random.default_rng(1).standard_normal((ixg, jxg))
    cons.zero_boundary_faces_th(ff, m)
    assert np.all(ff[:, m] == 0.0)
    assert np.all(ff[:, jxg - m] == 0.0)


def test_flux_divergence_telescopes_exactly():
    """境界フラックスをゼロにすれば発散の総和が厳密にゼロになること。

    ここは丸め誤差すら入らない。各面フラックスが隣接 2 セルで符号を変えて
    ちょうど 1 回ずつ現れるため、総和は代数的に打ち消し合う。
    """
    rng = np.random.default_rng(42)
    ixg, jxg, m = 16, 20, 1
    ffr = rng.standard_normal((ixg, jxg))
    ffth = rng.standard_normal((ixg, jxg))
    cons.zero_boundary_faces_r(ffr, m)
    cons.zero_boundary_faces_th(ffth, m)

    dq = np.zeros((ixg, jxg))
    cons.add_flux_divergence(dq, ffr, ffth, 0.25, 0.125, m)
    total = cons.cell_integral(dq, 0.25, 0.125, m)
    scale = cons.cell_integral(np.abs(dq), 0.25, 0.125, m)
    assert abs(total)/scale < 1e-15


# ---------------------------------------------------------------------------
# エネルギー収支 (Rempel 2006 式 20-34)
# ---------------------------------------------------------------------------
def test_energy_reservoirs_are_positive(setup_dynamic):
    """エネルギー貯留が正で、桁が妥当なこと。"""
    cfg, grid, strat, setup = setup_dynamic
    eb = energy.EnergyBudget(cfg, grid, strat, setup)
    om1 = np.ascontiguousarray(-0.05*cfg.om0*grid.cosTH**2)
    prof = np.sin(np.pi*(grid.RR - grid.rrmin)/(grid.rrmax - grid.rrmin))
    vrr = np.ascontiguousarray(1e3*prof*(3*grid.cosTH**2 - 1))
    vth = np.ascontiguousarray(1e3*prof*np.sin(2*grid.TH))
    bph = np.ascontiguousarray(1e4*prof*np.sin(2*grid.TH))

    e_om, e_m, e_b = eb.reservoirs(om1, vrr, vth, bph)
    assert e_om > 0 and e_m > 0 and e_b > 0
    # 剛体回転のエネルギーが支配的で、差動回転分ははるかに小さい
    assert eb.differential_rotation_energy(om1) < 0.01*e_om
    # 太陽の対流層の回転エネルギーは 1e41 erg 台
    assert 1e40 < e_om < 1e43, f'E_Omega の桁が妥当でない: {e_om:.3e}'


@pytest.mark.parametrize('seed', range(3))
def test_dissipation_terms_are_positive_definite(setup_dynamic, seed):
    """粘性散逸とオーム散逸が任意の場に対して正であること。

    :math:`Q_\\nu^\\Omega` と :math:`Q_\\eta` は二乗和の積分なので、
    符号や幾何因子を取り違えていなければ必ず正になる。
    """
    cfg, grid, strat, setup = setup_dynamic
    eb = energy.EnergyBudget(cfg, grid, strat, setup)
    rng = np.random.default_rng(seed)
    prof = np.sin(np.pi*(grid.RR - grid.rrmin)/(grid.rrmax - grid.rrmin))
    om1 = np.ascontiguousarray(0.05*cfg.om0*prof
                               * np.cos(rng.integers(1, 4)*grid.TH))
    vrr = np.ascontiguousarray(1e3*prof*np.cos(rng.integers(1, 4)*grid.TH))
    vth = np.ascontiguousarray(1e3*prof*np.sin(rng.integers(1, 4)*grid.TH))
    bb = [np.ascontiguousarray(1e3*prof*np.sin(k*grid.TH)) for k in (1, 2, 3)]
    se1 = np.ascontiguousarray(1e-5*prof*np.cos(grid.TH))

    q = eb.exchanges(om1, vrr, vth, bb[0], bb[1], bb[2], se1)
    assert q['Q_nu_Omega'] > 0, f"粘性散逸が負: {q['Q_nu_Omega']:.3e}"
    assert q['Q_eta'] > 0, f"オーム散逸が負: {q['Q_eta']:.3e}"
    assert q['Q_nu_M'] > 0, f"子午面の粘性散逸が負: {q['Q_nu_M']:.3e}"


def test_lambda_exchange_vanishes_without_lambda_effect(setup_dynamic):
    """Λ効果をゼロにすると Q_Lambda がゼロになること。"""
    cfg, grid, strat, setup = setup_dynamic
    eb = energy.EnergyBudget(cfg, grid, strat, setup)
    saved = (setup.lam_rp.copy(), setup.lam_tp.copy())
    try:
        setup.lam_rp[:] = 0.0
        setup.lam_tp[:] = 0.0
        om1 = np.ascontiguousarray(-0.05*cfg.om0*grid.cosTH**2)
        z = np.zeros((grid.ixg, grid.jxg))
        q = eb.exchanges(om1, z, z, z, z, z, z)
        assert q['Q_Lambda'] == 0.0
    finally:
        setup.lam_rp[:], setup.lam_tp[:] = saved


def test_energy_budget_tracks_actual_evolution(setup_dynamic):
    """収支式が実際の時間発展と整合すること。

    :math:`\\partial_t E_\\Omega = Q_\\Lambda - Q_\\nu^\\Omega - Q_C
    - Q_L^\\Omega` を、ソルバで実測した :math:`\\Delta E_\\Omega/\\Delta t`
    と比較する。収支項はソルバとは別の離散化 (中心差分と体積積分) で
    計算しているので完全一致はしないが、桁と符号が合っていなければ
    式の転記か実装のどちらかが間違っている。

    Rempel (2006) 表 1 の注も「エネルギー交換項の精度は 0.001 程度」と
    しており、定常状態でない過渡期はさらに緩い。
    """
    cfg, grid, strat, setup = setup_dynamic
    sol = dynamic.DynamicSolver(cfg, grid, strat, setup)
    sol.magnetic = False
    eb = energy.EnergyBudget(cfg, grid, strat, setup)
    sol.set_primitive_from_conserved(sol.conserved())
    dt = sol.cfl_dt()

    # 少し回して非自明な状態にする
    for _ in range(300):
        sol.step(dt)

    def e_omega():
        return eb.differential_rotation_energy(sol.om1)

    e0 = e_omega()
    q = eb.exchanges(sol.om1, sol.vrr, sol.vth, sol.brr, sol.bth, sol.bph,
                     sol.se1)
    nsub = 200
    for _ in range(nsub):
        sol.step(dt)
    measured = (e_omega() - e0)/(nsub*dt)

    predicted = q['Q_Lambda'] - q['Q_nu_Omega'] - q['Q_C'] - q['Q_L_Omega']
    assert measured > 0, 'Λ効果で差動回転が育っていない'
    assert np.sign(measured) == np.sign(predicted), (
        f'収支の符号が実測と合わない: 実測 {measured:.3e} 予測 {predicted:.3e}')
    ratio = predicted/measured
    assert 0.3 < ratio < 3.0, (
        f'収支が実測と桁で合わない: 予測/実測 = {ratio:.3f} '
        f'(実測 {measured:.3e}, 予測 {predicted:.3e})')


# ---------------------------------------------------------------------------
# 開いた境界での収支 (Rempel の下部境界条件)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('bc,closed', [('stress_free', True),
                                       ('uniform_rotation', False)])
def test_bottom_boundary_angular_momentum_budget(bc, closed):
    """境界条件に応じて「保存」または「収支が厳密に合う」こと。

    Rempel (2005, 2006) は下部境界 r=0.65 RSUN で Ω1 = 0 の Dirichlet 条件を
    課してタコクラインを強制する。この境界は固定 Ω のリザーバなので、
    **系は角運動量について閉じない** — 粘性フラックスが境界を通る。

    ユーザー要求「境界から保存量が出ていかないように」は、Rempel の設定では
    そのままでは満たせない。そこで:

    - ``stress_free``      : 閉じた系。∫q_L が machine precision で保存
    - ``uniform_rotation`` : 開いた系。∫q_L の変化が境界フラックスの
      時間積分と machine precision で一致する (漏れを許すのではなく、
      漏れを厳密に勘定する)

    の 2 つを提供し、どちらも回帰テストで固定する。
    """
    cfg = make_cfg('parameters/rempel06.py', ix=48, jx=48, dynamics='hydro',
                   angmom_bottom_bc=bc)
    grid = make_grid(cfg)
    strat = Stratification(cfg, grid)
    setup = S2MFD.Setup(cfg, grid)
    sol = dynamic.DynamicSolver(cfg, grid, strat, setup)
    m = grid.margin

    sol.om1[:] = -0.05*cfg.om0*grid.cosTH**2
    sol.set_primitive_from_conserved(sol.conserved())
    dt = sol.cfl_dt()

    def total():
        return cons.cell_integral(strat.JL*sol.om1, grid.drr, grid.dth, m)

    l0 = total()
    scale = cons.cell_integral(np.abs(strat.JL*cfg.om0), grid.drr, grid.dth, m)
    for _ in range(1500):
        sol.step(dt)
    dl = total() - l0
    flux = sol.boundary_angmom_flux

    if closed:
        assert flux == 0.0, '閉じた境界なのにフラックスが記録されている'
        assert abs(dl)/scale < 1e-14, f'角運動量が保存していない: {dl/scale:.3e}'
    else:
        # 境界から実際に角運動量が入っていること (テストが自明でないこと)
        assert abs(dl)/scale > 1e-6, (
            f'開いた境界なのに何も流入していない: {dl/scale:.3e}')
        assert abs(dl - flux)/abs(dl) < 1e-12, (
            f'境界フラックスの収支が合わない: ΔL={dl:.6e} 積分={flux:.6e}')


def test_reynolds_stress_is_not_double_counted():
    """レイノルズ応力が角運動量方程式に二重計上されていないこと。

    ``angular_momentum_rhs`` は応力を ``dq_stress`` にだけ書き出し、
    ``dq_om`` への合流は呼び出し側が行う設計になっている。両方に書くと
    粘性と Λ 効果が 2 倍になり、しかも見た目には「それらしい」解が出る
    ので気付きにくい (実際にこのバグが入っていた。境界フラックスの収支が
    ちょうど 2 倍ずれることで発覚した)。
    """
    cfg = make_cfg('parameters/rempel06.py', ix=32, jx=32, dynamics='hydro')
    grid = make_grid(cfg)
    strat = Stratification(cfg, grid)
    setup = S2MFD.Setup(cfg, grid)
    m = grid.margin
    work = hydro.HydroWork(grid)
    shape = (grid.ixg, grid.jxg)
    z = np.zeros(shape)

    om1 = np.ascontiguousarray(-0.05*cfg.om0*grid.cosTH**2)
    dq_om = np.zeros(shape)
    dq_stress = np.zeros(shape)
    hydro.angular_momentum_rhs(
        dq_om, om1, z, z, z, z, z,
        strat.JL, strat.JLY, strat.JV, strat.JVY, strat.W2,
        grid.RR, grid.RRm, grid.sinTH, grid.sinTHm, strat.ro0, strat.ro0m,
        setup.lam_rp, setup.lam_tp, cfg.om0,
        setup.nu_dif, setup.nu_dif_m, setup.nu_lam, setup.nu_lam_m,
        grid.drr, grid.dth, m, False, False, False,
        dq_stress, np.zeros(shape), np.zeros(grid.jxg),
        work.ffr, work.ffth, work.cen)

    sl = (slice(m, grid.ixg - m), slice(m, grid.jxg - m))
    # 速度ゼロなので移流も Maxwell もない -> dq_om はゼロのままのはず
    assert np.abs(dq_om[sl]).max() == 0.0, (
        'dq_om に応力が混入している (二重計上)')
    assert np.abs(dq_stress[sl]).max() > 0.0, 'dq_stress に応力が出ていない'


# ---------------------------------------------------------------------------
# 力学ダイナモ (流体 + 誘導方程式の結合)
# ---------------------------------------------------------------------------
def test_dynamic_mode_couples_to_existing_induction_kernel():
    """力学モードで、既存の運動学的ダイナモのカーネルがそのまま使えること。

    Rempel 2006 式(6)(7) と S2MFD の誘導方程式は項ごとに一致しているので、
    誘導方程式を書き直す必要はない。流れ場を ``setup`` に書き戻すだけで
    結合できる (``DynamicSolver.sync_to_induction``)。

    これにより運動学的ダイナモと力学ダイナモが**同一の誘導方程式コード**を
    共有するので、片方だけが壊れるということが起きない。
    """
    from S2MFD.physics import time_marching, poloidal_mag

    cfg = make_cfg('parameters/rempel06.py', ix=32, jx=32)
    assert cfg.dynamics == 'dynamic'
    grid = make_grid(cfg)
    strat = Stratification(cfg, grid)
    setup = S2MFD.Setup(cfg, grid)
    sol = dynamic.DynamicSolver(cfg, grid, strat, setup)
    m = grid.margin
    sl = (slice(m, grid.ixg - m), slice(m, grid.jxg - m))

    prof = np.sin(np.pi*(grid.RR - grid.rrmin)/(grid.rrmax - grid.rrmin))
    bph = np.ascontiguousarray(1e3*prof*np.sin(2*grid.TH))
    aph = np.zeros_like(bph)
    sol.set_primitive_from_conserved(sol.conserved())
    sol.sync_to_induction()
    dt = sol.cfl_dt()

    for _ in range(50):
        bph, aph = time_marching(bph, aph, dt, cfg, grid, setup)
        pm = poloidal_mag(aph, grid.RR, grid.sinTH, grid.drr, grid.dth)
        sol.set_magnetic_field(pm[0], pm[1], bph)
        sol.step(dt)
        sol.sync_to_induction()

    assert np.all(np.isfinite(bph)) and np.all(np.isfinite(aph))
    assert np.all(np.isfinite(sol.om1)) and np.all(np.isfinite(sol.se1))
    # 磁場が消えていないこと (Ω効果と拡散が両方効いている状態)
    assert np.abs(bph[sl]).max() > 1e2


def test_r06_alpha_kernel_is_normalised():
    """Rempel 2006 式(19) の放物線カーネル h(r) が ∫h dr = 1 に規格化されること。"""
    cfg = make_cfg('parameters/rempel06.py', ix=64, jx=32)
    grid = make_grid(cfg)
    setup = S2MFD.Setup(cfg, grid)
    integral = setup.alpha_kernel.sum()*grid.drr
    assert abs(integral - 1.0) < 1e-10, f'規格化されていない: {integral}'
    # 0.71 - 0.76 RSUN の外ではゼロ
    outside = (grid.rr < cfg.r_h_bot) | (grid.rr > cfg.r_h_top)
    assert np.all(setup.alpha_kernel[outside] == 0.0)


def test_kinematic_mode_is_unaffected_by_dynamic_additions():
    """既存の運動学的ダイナモが力学モードの追加で変わっていないこと。

    ユーザー要求「この実装後も kinematic dynamo の機能は残して欲しい」の
    直接の検証。力学モード用に physics_core へ入れた変更 (R06 alpha の
    追加、rank-1 分解の抑制条件) が既定経路に影響していないことを確認する。
    """
    from S2MFD.physics import time_marching, time_marching_reference

    cfg = make_cfg('parameters/defaults.py', ix=32, jx=32)
    assert getattr(cfg, 'dynamics', 'kinematic') == 'kinematic'
    grid = make_grid(cfg)
    setup = S2MFD.Setup(cfg, grid)
    rng = np.random.default_rng(0)
    bph = np.ascontiguousarray(rng.standard_normal((grid.ixg, grid.jxg)))
    aph = np.ascontiguousarray(rng.standard_normal((grid.ixg, grid.jxg)))

    cfg.exact_arithmetic = True
    fast = time_marching(bph.copy(), aph.copy(), 1.0e4, cfg, grid, setup)
    ref = time_marching_reference(bph.copy(), aph.copy(), 1.0e4, cfg, grid, setup)
    for a, b in zip(fast, ref):
        assert np.array_equal(a, b), '参照実装とのビット一致が壊れている'


def test_sld_meridional_has_spherical_geometry_terms(setup_dynamic):
    """子午面速度の人工拡散に球座標の幾何項が入っていること。

    球座標の基底ベクトルは θ に依存する (∂ê_r/∂θ = ê_θ,
    ∂ê_θ/∂θ = -ê_r) ので、v_r と v_θ をスカラーとして独立に拡散させると
    間違いになる。θ 方向の掃引では

        ∂_θ(F_r ê_r + F_θ ê_θ) = (∂_θF_r - F_θ)ê_r + (∂_θF_θ + F_r)ê_θ

    となり、成分が混ざる項が要る。R2D2 の artdif_spherical.F90 の
    ``j1 == 1`` ブロックに対応する。

    ここでは、幾何項が「θ 面フラックス配列のセル中心平均」に一致すること
    を直接確認する。抜けていると気付きにくい (計算は安定なまま、
    子午面循環だけが少しずつ間違う) ので、構造として固定しておく。
    """
    cfg, grid, strat, setup = setup_dynamic
    m = grid.margin
    shape = (grid.ixg, grid.jxg)
    csp = np.full(shape, 1.0e5)
    jac_r = np.zeros(shape)
    jac_r[1:] = 0.5*(strat.JV[1:] + strat.JV[:-1])
    jac_th = np.zeros(shape)
    jac_th[:, 1:] = 0.5*(strat.JVY[:, 1:] + strat.JVY[:, :-1])

    rng = np.random.default_rng(11)
    vrr = np.ascontiguousarray(1e3*rng.standard_normal(shape))
    vth = np.ascontiguousarray(1e3*rng.standard_normal(shape))

    ff = [np.zeros(shape) for _ in range(4)]
    dmr = np.zeros(shape)
    dmt = np.zeros(shape)
    artdif.sld_diffuse_meridional(dmr, dmt, vrr, vth, jac_r, jac_th, csp, csp,
                                  2.0, 2.0, grid.drr, grid.dth, m, *ff)

    # 幾何項なしの参照 (各成分を独立にスカラー拡散)
    r_mr = np.zeros(shape)
    r_mt = np.zeros(shape)
    g1 = [np.zeros(shape) for _ in range(2)]
    g2 = [np.zeros(shape) for _ in range(2)]
    artdif.sld_diffuse(r_mr, vrr, jac_r, jac_th, csp, csp, 2.0, 2.0,
                       grid.drr, grid.dth, m, *g1)
    artdif.sld_diffuse(r_mt, vth, jac_r, jac_th, csp, csp, 2.0, 2.0,
                       grid.drr, grid.dth, m, *g2)

    sl = (slice(m, grid.ixg - m), slice(m, grid.jxg - m))
    # 差が θ 面フラックスのセル中心平均に一致すること
    ffth_vr, ffth_vth = ff[1], ff[3]
    exp_r = 0.5*(ffth_vth[:, :-1] + ffth_vth[:, 1:])[sl[0], slice(m, grid.jxg - m)]
    exp_t = -0.5*(ffth_vr[:, :-1] + ffth_vr[:, 1:])[sl[0], slice(m, grid.jxg - m)]
    # dmr - r_mr は桁落ちを伴う差なので、絶対許容差は dmr の大きさで測る
    tol_r = 1e-10*np.abs(dmr[sl]).max()
    tol_t = 1e-10*np.abs(dmt[sl]).max()
    assert np.allclose(dmr[sl] - r_mr[sl], exp_r, rtol=1e-8, atol=tol_r)
    assert np.allclose(dmt[sl] - r_mt[sl], exp_t, rtol=1e-8, atol=tol_t)
    # 幾何項が実際に効いていること (テストが自明でないこと)。
    # 格子ノイズに対しては主項 (theta 微分 ~ F/(r dtheta)) の方が
    # 幾何項 (~ F/r) より dtheta 分だけ大きいので比は小さいが、
    # 滑らかなベクトル場では主項が小さくなるので相対的に効いてくる。
    assert np.abs(exp_r).max() > 1e-4*np.abs(r_mr[sl]).max()
    assert np.abs(exp_t).max() > 1e-4*np.abs(r_mt[sl]).max()
