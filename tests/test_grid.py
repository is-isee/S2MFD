"""Grid クラスのテスト(座標生成と save/load)。"""
import numpy as np
import pytest

import S2MFD


def _grid(ix=16, jx=8, margin=1):
    return S2MFD.Grid(ix=ix, jx=jx, margin=margin,
                      rrmin=0.65, rrmax=1.0, thmin=0.0, thmax=np.pi)


class TestGridGeometry:
    def test_sizes_and_spacing(self):
        g = _grid()
        assert g.ixg == 16 + 2
        assert g.jxg == 8 + 2
        assert np.isclose(g.drr, 0.35 / 16)
        assert np.isclose(g.dth, np.pi / 8)

    def test_cell_centered_coordinates(self):
        g = _grid()
        # 最初の物理セル (margin=1 → index 1) はセル中心 rrmin + drr/2
        assert np.isclose(g.rr[g.margin], 0.65 + g.drr * 0.5)
        assert np.isclose(g.th[g.margin], g.dth * 0.5)
        # ゴーストセルは領域外に対称に配置
        assert np.isclose(g.rr[0], 0.65 - g.drr * 0.5)
        # 等間隔
        assert np.allclose(np.diff(g.rr), g.drr)
        assert np.allclose(np.diff(g.th), g.dth)

    def test_face_centered_coordinates(self):
        g = _grid()
        assert np.allclose(g.RRm[1:, :], 0.5 * (g.RR[1:, :] + g.RR[:-1, :]))
        assert np.allclose(g.THm[:, 1:], 0.5 * (g.TH[:, 1:] + g.TH[:, :-1]))
        # RRm[0,:] / THm[:,0] は未定義 (0 のまま) — 現実装の仕様
        assert np.allclose(g.RRm[0, :], 0.0)

    def test_meshgrid_orientation(self):
        g = _grid()
        # indexing='ij': 第0軸が r、第1軸が θ
        assert g.RR.shape == (g.ixg, g.jxg)
        assert np.allclose(g.RR[:, 0], g.rr)
        assert np.allclose(g.TH[0, :], g.th)


class TestGridSaveLoad:
    def test_arrays_roundtrip(self, tmp_path):
        g = _grid()
        path = str(tmp_path / 'grid.npz')
        g.save(path)
        loaded = S2MFD.Grid.load(path)
        assert np.allclose(loaded.rr, g.rr)
        assert np.allclose(loaded.RR, g.RR)
        assert np.allclose(loaded.sinTH, g.sinTH)

    def test_scalars_keep_python_types(self, tmp_path):
        # Phase 2 で修正済み: NpzIO.load が0次元配列を .item() で復元する
        g = _grid()
        path = str(tmp_path / 'grid.npz')
        g.save(path)
        loaded = S2MFD.Grid.load(path)
        assert isinstance(loaded.margin, (int, np.integer)) and np.ndim(loaded.margin) == 0
        assert not isinstance(loaded.margin, np.ndarray)
        assert not isinstance(loaded.drr, np.ndarray)
        assert not isinstance(loaded.ixg, np.ndarray)
