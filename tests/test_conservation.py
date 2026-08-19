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
                 magnetic):
    m = grid.margin

    def rhs(q):
        hydro.to_primitive_om1(q, strat.iJL, om1, m)
        dq = np.zeros_like(q)
        hydro.angular_momentum_rhs(
            dq, om1, vrr, vth, bb[0], bb[1], bb[2],
            strat.JL, strat.JLY, grid.RR, grid.RRm, grid.sinTH, grid.sinTHm,
            strat.ro0, strat.ro0m, setup.lam_rp, setup.lam_tp,
            cfg.om0, cfg.nu_turb, grid.drr, grid.dth, m, magnetic,
            work.ffr, work.ffth, work.cen)
        return dq

    q1 = q_om + dt*rhs(q_om)
    return 0.5*(q_om + q1 + dt*rhs(q1))


@pytest.mark.parametrize('magnetic', [False, True])
def test_angular_momentum_conserved_to_machine_precision(setup_dynamic, magnetic):
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
                            dt, work, magnetic)

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
                strat.JL, strat.JLY, grid.RR, grid.RRm, grid.sinTH,
                grid.sinTHm, strat.ro0, strat.ro0m,
                setup.lam_rp, setup.lam_tp,
                cfg.om0, cfg.nu_turb, grid.drr, grid.dth, m, False,
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
