"""時間積分ループを numba 側に閉じ込めるための実装。

Python から substep ごとに numba カーネルを呼ぶと、引数 20 個超の型検査
(ディスパッチ) が実計算より重くなる。ここでは
「境界条件 → RK2 の 1 ステップ → 出力時刻までのループ」までを
まとめて 1 つの njit 関数にし、Python 側の呼び出しを出力間隔ごと
(既定 40 日 ≒ 24 ステップ) の 1 回に減らす。
"""
import numpy as np
from numba import njit

from .physics_core import (get_time_marching_kernel, separable_profiles,
                           ALPHA_CODE)

BC_VERTICAL = 0
BC_POTENTIAL = 1
BC_CODE = {'vertical': BC_VERTICAL, 'potential': BC_POTENTIAL}


def _make_bc(bc_code):
    """境界条件を njit 化する (種別はコンパイル時に固定)。"""

    @njit(boundscheck=False)
    def bc(Bph, Aph, rr, margin, potential_operator):
        ixg, jxg = Bph.shape
        jlo = margin
        jhi = jxg - margin

        if bc_code == BC_POTENTIAL:
            # 表面値 → ゴースト値の線形作用素 (Legendre 射影と再構成を合成済み)
            surf = np.ascontiguousarray(Aph[ixg - margin - 1, jlo:jhi])

        for i in range(margin):
            # 上部境界: Bph = 0 (反対称)
            for j in range(jlo, jhi):
                Bph[ixg-i-1, j] = -Bph[ixg-2*margin+i, j]
            if bc_code == BC_POTENTIAL:
                # 内側ループを jp (連続方向) にして、出力へ加算していく
                # (j を内側にすると op[j, jp] のストライドが jx になり遅い)
                op = potential_operator[i]
                nj = jhi - jlo
                row = Aph[ixg-i-1]
                for jp in range(nj):
                    row[jlo+jp] = 0.0
                for j in range(nj):
                    sj = surf[j]
                    for jp in range(nj):
                        row[jlo+jp] += sj*op[j, jp]
            else:
                # d(r*Aph)/dr = 0
                for j in range(jlo, jhi):
                    Aph[ixg-i-1, j] = (Aph[ixg-2*margin+i, j]
                                       / rr[ixg-i-1]*rr[ixg-2*margin+i])
            # 下部境界: 完全導体 (Aph = 0, d(r*Bph)/dr = 0)
            for j in range(jlo, jhi):
                Aph[i, j] = -Aph[2*margin-i-1, j]
                Bph[i, j] = Bph[2*margin-i-1, j]/rr[i]*rr[2*margin-i-1]

        # 極 (θ=0, π): 反対称
        for j in range(margin):
            for i in range(margin, ixg - margin):
                Bph[i, j] = -Bph[i, 2*margin-j-1]
                Aph[i, j] = -Aph[i, 2*margin-j-1]
                Bph[i, jxg-j-1] = -Bph[i, jxg-2*margin+j]
                Aph[i, jxg-j-1] = -Aph[i, jxg-2*margin+j]
        return Bph, Aph

    return bc


def _make_advance(fast, nonlocal_alpha, bc_code, separable):
    """出力時刻まで進める njit ループを生成する。"""
    march = get_time_marching_kernel(fast, nonlocal_alpha, separable)
    bc = _make_bc(bc_code)

    @njit(boundscheck=False)
    def advance(Bph, Aph, time, dt, t_stop, rr, sth, rrm, sthm, drr, dth,
                urr, uth, et, etrr, omrr, omth, so, ibase, alpha_code,
                inv_rr, inv_rr2, inv_sth, inv_sth2, margin, potential_operator,
                urr_u, urr_v, uth_u, uth_v, et_u, et_v, etrr_u, etrr_v,
                omth_u, omth_v, so_u, so_v):
        n = 0
        # 作業配列はループの外で1度だけ確保して使い回す
        Bphm = np.empty_like(Bph)
        Aphm = np.empty_like(Aph)
        Bphn = np.empty_like(Bph)
        Aphn = np.empty_like(Aph)
        alpha_fac = np.zeros(Bph.shape[1])
        while time < t_stop:
            # 非局所 alpha の係数は Bph が変わるたびに更新する
            if nonlocal_alpha:
                for j in range(Bph.shape[1]):
                    b = Bph[ibase, j]
                    alpha_fac[j] = b/(1 + b**2)

            march(Bph, Aph, dt, rr, sth, rrm, sthm, drr, dth,
                               urr, uth, et, etrr, omrr, omth, so, ibase,
                  alpha_code, inv_rr, inv_rr2, inv_sth, inv_sth2,
                  alpha_fac, Bphm, Aphm,
                  urr_u, urr_v, uth_u, uth_v, et_u, et_v, etrr_u, etrr_v,
                  omth_u, omth_v, so_u, so_v)
            bc(Bphm, Aphm, rr, margin, potential_operator)

            if nonlocal_alpha:
                for j in range(Bph.shape[1]):
                    b = Bphm[ibase, j]
                    alpha_fac[j] = b/(1 + b**2)
            march(Bphm, Aphm, dt, rr, sth, rrm, sthm, drr, dth,
                  urr, uth, et, etrr, omrr, omth, so, ibase,
                  alpha_code, inv_rr, inv_rr2, inv_sth, inv_sth2,
                  alpha_fac, Bphn, Aphn,
                  urr_u, urr_v, uth_u, uth_v, et_u, et_v, etrr_u, etrr_v,
                  omth_u, omth_v, so_u, so_v)
            bc(Bphn, Aphn, rr, margin, potential_operator)

            for i in range(Bph.shape[0]):
                for j in range(Bph.shape[1]):
                    Bph[i, j] = 0.5*Bph[i, j] + 0.5*Bphn[i, j]
                    Aph[i, j] = 0.5*Aph[i, j] + 0.5*Aphn[i, j]

            time += dt
            n += 1
        return Bph, Aph, time, n

    return advance


_ADVANCE = {}


def get_advance(fast, nonlocal_alpha, bc_code, separable):
    key = (fast, nonlocal_alpha, bc_code, separable)
    fn = _ADVANCE.get(key)
    if fn is None:
        fn = _make_advance(*key)
        _ADVANCE[key] = fn
    return fn


def advance_to(sim, t_stop):
    """sim を t_stop まで進める (numba ループを 1 回呼ぶ)。

    Parameters
    ----------
    sim : S2MFD.Simulation
    t_stop : float
        この時刻に達するまで進める

    Returns
    -------
    int
        実行したステップ数
    """
    from .physics_core import _grid_1d
    cfg, grid, setup = sim.cfg, sim.grid, sim.setup
    alpha_code = ALPHA_CODE.get(cfg.alpha_type)
    if alpha_code is None:
        raise ValueError(f"unknown alpha_type: {cfg.alpha_type!r}")
    bc_code = BC_CODE.get(cfg.boundary_condition_type)
    if bc_code is None:
        raise ValueError("unknown boundary_condition_type: "
                         f"{cfg.boundary_condition_type!r}")
    rr, sth, rrm, sthm, inv_rr, inv_rr2, inv_sth, inv_sth2 = _grid_1d(grid)
    fast = not getattr(cfg, 'exact_arithmetic', False)
    pot_op = getattr(sim.legendre, 'potential_operator', None)
    if pot_op is None:
        pot_op = np.zeros((grid.margin, grid.jx, grid.jx))

    sep, factors = separable_profiles(setup)
    sep = sep and fast
    fn = get_advance(fast, alpha_code == 0, bc_code, sep)
    Bph, Aph, time, n = fn(
        sim.Bph, sim.Aph, float(sim.time), float(sim.dt), float(t_stop),
        rr, sth, rrm, sthm, grid.drr, grid.dth,
        setup.urr, setup.uth, setup.et, setup.etrr,
        setup.omrr, setup.omth, setup.so, setup.ibase, alpha_code,
        inv_rr, inv_rr2, inv_sth, inv_sth2, grid.margin, pot_op, *factors)
    sim.Bph = Bph
    sim.Aph = Aph
    sim.time = time
    sim.n += n
    return n


def _make_advance_min(fast, nonlocal_alpha, bc_code, separable):
    """黒点数プロキシの極小が現れるまで進める njit ループを生成する。"""
    march = get_time_marching_kernel(fast, nonlocal_alpha, separable)
    bc = _make_bc(bc_code)

    @njit(boundscheck=False)
    def advance_min(Bph, Aph, time, dt, t_limit, rr, sth, rrm, sthm, drr, dth,
                    urr, uth, et, etrr, omrr, omth, so, ibase, alpha_code,
                    inv_rr, inv_rr2, inv_sth, inv_sth2, margin,
                    potential_operator, base, loca, gamma, sn0, sn1, sn2,
                    urr_u, urr_v, uth_u, uth_v, et_u, et_v, etrr_u, etrr_v,
                    omth_u, omth_v, so_u, so_v):
        n = 0
        ok = True
        # 作業配列はループの外で1度だけ確保して使い回す
        Bphm = np.empty_like(Bph)
        Aphm = np.empty_like(Aph)
        Bphn = np.empty_like(Bph)
        Aphn = np.empty_like(Aph)
        alpha_fac = np.zeros(Bph.shape[1])
        while not (sn1 < sn0 and sn1 < sn2):
            if time > t_limit:
                ok = False
                break
            if nonlocal_alpha:
                for j in range(Bph.shape[1]):
                    b = Bph[ibase, j]
                    alpha_fac[j] = b/(1 + b**2)
            march(Bph, Aph, dt, rr, sth, rrm, sthm, drr, dth,
                  urr, uth, et, etrr, omrr, omth, so, ibase,
                  alpha_code, inv_rr, inv_rr2, inv_sth, inv_sth2,
                  alpha_fac, Bphm, Aphm,
                  urr_u, urr_v, uth_u, uth_v, et_u, et_v, etrr_u, etrr_v,
                  omth_u, omth_v, so_u, so_v)
            bc(Bphm, Aphm, rr, margin, potential_operator)
            if nonlocal_alpha:
                for j in range(Bph.shape[1]):
                    b = Bphm[ibase, j]
                    alpha_fac[j] = b/(1 + b**2)
            march(Bphm, Aphm, dt, rr, sth, rrm, sthm, drr, dth,
                  urr, uth, et, etrr, omrr, omth, so, ibase,
                  alpha_code, inv_rr, inv_rr2, inv_sth, inv_sth2,
                  alpha_fac, Bphn, Aphn,
                  urr_u, urr_v, uth_u, uth_v, et_u, et_v, etrr_u, etrr_v,
                  omth_u, omth_v, so_u, so_v)
            bc(Bphn, Aphn, rr, margin, potential_operator)
            for i in range(Bph.shape[0]):
                for j in range(Bph.shape[1]):
                    Bph[i, j] = 0.5*Bph[i, j] + 0.5*Bphn[i, j]
                    Aph[i, j] = 0.5*Aph[i, j] + 0.5*Aphn[i, j]
            time += dt
            n += 1
            sn0 = sn1
            sn1 = sn2
            sn2 = gamma*Bph[base, loca]**2
        return Bph, Aph, time, n, ok, sn0, sn1, sn2

    return advance_min


_ADVANCE_MIN = {}


def _kernel_keys(sim):
    cfg = sim.cfg
    alpha_code = ALPHA_CODE.get(cfg.alpha_type)
    if alpha_code is None:
        raise ValueError(f"unknown alpha_type: {cfg.alpha_type!r}")
    bc_code = BC_CODE.get(cfg.boundary_condition_type)
    if bc_code is None:
        raise ValueError("unknown boundary_condition_type: "
                         f"{cfg.boundary_condition_type!r}")
    fast = not getattr(cfg, 'exact_arithmetic', False)
    return fast, alpha_code, bc_code


def _pack_args(sim):
    from .physics_core import _grid_1d
    grid, setup = sim.grid, sim.setup
    rr, sth, rrm, sthm, inv_rr, inv_rr2, inv_sth, inv_sth2 = _grid_1d(grid)
    pot_op = getattr(sim.legendre, 'potential_operator', None)
    if pot_op is None:
        pot_op = np.zeros((grid.margin, grid.jx, grid.jx))
    return (rr, sth, rrm, sthm, grid.drr, grid.dth,
            setup.urr, setup.uth, setup.et, setup.etrr,
            setup.omrr, setup.omth, setup.so, setup.ibase,
            inv_rr, inv_rr2, inv_sth, inv_sth2, grid.margin, pot_op)


def advance_until_minimum(sim, t_limit, sn_history, base, loca, gamma):
    """黒点数プロキシの極小が現れるまで進める。

    Returns
    -------
    tuple
        (ステップ数, 検出できたか)
    """
    fast, alpha_code, bc_code = _kernel_keys(sim)
    sep, factors = separable_profiles(sim.setup)
    sep = sep and fast
    key = (fast, alpha_code == 0, bc_code, sep)
    fn = _ADVANCE_MIN.get(key)
    if fn is None:
        fn = _make_advance_min(*key)
        _ADVANCE_MIN[key] = fn
    (rr, sth, rrm, sthm, drr, dth, urr, uth, et, etrr, omrr, omth, so, ibase,
     inv_rr, inv_rr2, inv_sth, inv_sth2, margin, pot_op) = _pack_args(sim)
    Bph, Aph, time, n, ok, s0, s1, s2 = fn(
        sim.Bph, sim.Aph, float(sim.time), float(sim.dt), float(t_limit),
        rr, sth, rrm, sthm, drr, dth, urr, uth, et, etrr, omrr, omth, so,
        ibase, alpha_code, inv_rr, inv_rr2, inv_sth, inv_sth2, margin, pot_op,
        base, loca, gamma, sn_history[0], sn_history[1], sn_history[2],
        *factors)
    sim.Bph = Bph
    sim.Aph = Aph
    sim.time = time
    sim.n += n
    sn_history[0], sn_history[1], sn_history[2] = s0, s1, s2
    return n, ok
