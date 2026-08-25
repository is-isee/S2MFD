"""微分演算子 (S2MFD.tools) のテスト。

注意: drr1/dth1 の docstring は実装と逆である (2026-08-19 レビューで確認)。
実装の実際の挙動:
  - dir='up': dqq[i] = (qq[i] - qq[i-1]) / d   (後退差分、i = 1..N-1 に格納)
  - dir='dw': dqq[i] = (qq[i+1] - qq[i]) / d   (前進差分、i = 0..N-2 に格納)
diffusion() では 'up' でセル境界 i-1/2 の勾配を作り 'dw' で発散を取る
保存形ペアとして正しく使われている。本テストは実装の挙動を正として固定する。
"""
import numpy as np
import pytest

from conftest import make_cfg, make_grid

from S2MFD.tools import drr1, drr2, dth1, dth2


def _linear_r(nx=8, ny=5, a=2.0, b=1.0, d=0.1):
    r = np.arange(nx) * d
    qq = a * r[:, None] + b * np.ones((nx, ny))
    return qq, d


def _linear_th(nx=5, ny=8, a=2.0, b=1.0, d=0.1):
    t = np.arange(ny) * d
    qq = a * t[None, :] + b * np.ones((nx, ny))
    return qq, d


class TestDrr1:
    def test_up_is_backward_difference(self):
        qq, d = _linear_r()
        dqq = drr1(qq, np.full(qq.shape[0], d), 'up')
        # i=0 は 0 のまま、i>=1 に (qq[i]-qq[i-1])/d
        assert np.allclose(dqq[0, :], 0.0)
        assert np.allclose(dqq[1:, :], 2.0)

    def test_dw_is_forward_difference(self):
        qq, d = _linear_r()
        dqq = drr1(qq, np.full(qq.shape[0], d), 'dw')
        # i=N-1 は 0 のまま、i<=N-2 に (qq[i+1]-qq[i])/d
        assert np.allclose(dqq[-1, :], 0.0)
        assert np.allclose(dqq[:-1, :], 2.0)

    def test_up_dw_conservative_pair(self):
        """'up' → 'dw' の合成が2階中心差分(保存形)になること。"""
        nx, d = 32, 0.05
        r = np.arange(nx) * d
        qq = np.sin(r)[:, None] * np.ones((nx, 3))
        hh = np.full(nx, d)
        lap = drr1(drr1(qq, hh, 'up'), hh, 'dw')
        expected = (qq[2:, :] - 2 * qq[1:-1, :] + qq[:-2, :]) / d**2
        assert np.allclose(lap[1:-1, :], expected)


class TestDrr2:
    def test_exact_for_quadratic(self):
        nx, d = 16, 0.1
        r = np.arange(nx) * d
        qq = (3.0 * r**2 + 2.0 * r + 1.0)[:, None] * np.ones((nx, 4))
        dqq = drr2(qq, np.full(nx, 2.0*d))
        expected = (6.0 * r + 2.0)[1:-1]
        assert np.allclose(dqq[1:-1, :], expected[:, None])
        # 端は 0 のまま
        assert np.allclose(dqq[0, :], 0.0)
        assert np.allclose(dqq[-1, :], 0.0)

    def test_second_order_convergence(self):
        errors = []
        for nx in (32, 64):
            d = 1.0 / nx
            r = np.arange(nx) * d
            qq = np.sin(2 * np.pi * r)[:, None] * np.ones((nx, 2))
            dqq = drr2(qq, np.full(nx, 2.0*d))
            expected = 2 * np.pi * np.cos(2 * np.pi * r)
            errors.append(np.max(np.abs(dqq[1:-1, 0] - expected[1:-1])))
        # 格子を半分にすると誤差 ~1/4
        assert errors[1] < errors[0] / 3.0


class TestDth1:
    def test_up_is_backward_difference(self):
        qq, d = _linear_th()
        dqq = dth1(qq, d, 'up')
        assert np.allclose(dqq[:, 0], 0.0)
        assert np.allclose(dqq[:, 1:], 2.0)

    def test_dw_is_forward_difference(self):
        qq, d = _linear_th()
        dqq = dth1(qq, d, 'dw')
        assert np.allclose(dqq[:, -1], 0.0)
        assert np.allclose(dqq[:, :-1], 2.0)


class TestDth2:
    def test_exact_for_quadratic(self):
        ny, d = 16, 0.1
        t = np.arange(ny) * d
        qq = (t**2)[None, :] * np.ones((4, ny))
        dqq = dth2(qq, d)
        assert np.allclose(dqq[:, 1:-1], (2.0 * t)[None, 1:-1])


class TestAnaIndexHelpers:
    """``ana/ana_common.py`` の添字ヘルパがゴーストセルを踏まないこと。

    2026-08-23 以前の ``ana/`` は表面を ``Brrt[-2]`` で取っていた。これは
    **margin=1 でしか最外物理セルにならず**、margin=2 (Rempel 設定) では
    ゴーストセルを指す。半径も ``1+np.argmin(abs(grid.rr - r))`` で、
    ``grid.rr`` がゴースト込みなので +1 は 1 セル外側だった
    (0.7006 R のつもりが 0.7033 R)。

    どちらも「配列の端をゴースト込みで触る」型で、同じ型が診断側でも
    繰り返し出ている (``doc/dev_records/2026-08-23_mistakes.md`` の
    失敗パターン A)。
    """

    @staticmethod
    def _ana():
        import os
        import sys
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(root, 'ana')
        if path not in sys.path:
            sys.path.insert(0, path)
        import ana_common
        return ana_common

    @pytest.mark.parametrize('parameter_file,margin', [
        ('parameters/hotta10.py', 1),
        ('parameters/rempel06_paper.py', 2),
    ])
    def test_surface_index_is_a_physical_cell(self, parameter_file, margin):
        a = self._ana()
        cfg = make_cfg(parameter_file)
        grid = make_grid(cfg)
        assert grid.margin == margin
        i = a.surface_index(grid)
        assert grid.margin <= i < grid.ixg - grid.margin, (
            f"surface_index={i} が物理セル "
            f"[{grid.margin}, {grid.ixg-grid.margin}) の外")
        # 最外の物理セルであること
        assert i == grid.ixg - grid.margin - 1

    def test_surface_index_matches_old_minus_two_for_margin_one(self):
        """margin=1 では従来の ``[-2]`` と一致すること (図が変わらない)。"""
        a = self._ana()
        grid = make_grid(make_cfg('parameters/hotta10.py'))
        assert a.surface_index(grid) == grid.ixg - 2

    @pytest.mark.parametrize('parameter_file', [
        'parameters/hotta10.py', 'parameters/rempel06_paper.py'])
    def test_radial_index_stays_inside_and_is_nearest(self, parameter_file):
        a = self._ana()
        cfg = make_cfg(parameter_file)
        grid = make_grid(cfg)
        m = grid.margin
        for frac in (0.7, 0.735, 0.985):
            i = a.radial_index(grid, frac*cfg.RSUN)
            assert m <= i < grid.ixg - m
            inner = grid.rr[m:grid.ixg-m]
            best = np.min(np.abs(inner - frac*cfg.RSUN))
            assert abs(grid.rr[i] - frac*cfg.RSUN) == pytest.approx(best)

    def test_radial_index_is_not_the_old_off_by_one(self):
        """旧実装 ``1+argmin`` が 1 セルずれていたことを固定する。"""
        a = self._ana()
        cfg = make_cfg('parameters/rempel06_paper.py')
        grid = make_grid(cfg)
        i = a.radial_index(grid, 0.7*cfg.RSUN)
        old = 1 + int(np.argmin(np.abs(grid.rr - 0.7*cfg.RSUN)))
        assert old == i + 1, "旧実装との関係が変わった。docstring を見直すこと"
        assert abs(grid.rr[i] - 0.7*cfg.RSUN) < abs(grid.rr[old] - 0.7*cfg.RSUN)

    @pytest.mark.parametrize('parameter_file', [
        'parameters/hotta10.py', 'parameters/rempel06_paper.py'])
    def test_colat_index_stays_inside(self, parameter_file):
        a = self._ana()
        grid = make_grid(make_cfg(parameter_file))
        m = grid.margin
        for deg in (30.0, 60.0, 89.0):
            j = a.colat_index(grid, deg)
            assert m <= j < grid.jxg - m
