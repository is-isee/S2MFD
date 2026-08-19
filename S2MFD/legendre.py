import numpy as np

from S2MFD.npz_io import NpzIO


class Legendre(NpzIO):
    """
    Class for managing the associated Legendre polynomials used by the
    'potential' upper boundary condition.

    Attributes
    ----------
    costh : numpy.ndarray
        numpy.cos(th) on the interior (physical) colatitude grid
    lmax : int
        Maximum Legendre degree (= grid.jx)
    P1n : numpy.ndarray
        Associated Legendre polynomials P^1_n(cos th), shape (lmax, jx)
    sinth : numpy.ndarray
        numpy.sin(th) on the interior grid (precomputed for the BC)
    n_values : numpy.ndarray
        Degrees n = 1 .. lmax-1 (precomputed for the BC)
    coefficients : numpy.ndarray
        (2n+1) / (2n(n+1)) projection coefficients (precomputed for the BC)
    P1n_reduced : numpy.ndarray
        P1n without the unused n=0 row, shape (lmax-1, jx)
    """
    def __init__(self, grid):
        # 0<θ<πで定義
        self.costh = np.cos(grid.th[grid.margin:grid.jxg-grid.margin])
        self.lmax = grid.jx
        self.P1n = np.zeros((self.lmax, grid.jx))
        # n = 1
        self.P1n[1, :] = -(1-self.costh**2)**0.5
        # n = 2, 3, ...
        for i in range(2, self.lmax):
            # recurrence relation
            self.P1n[i, :] = ((2*i-1)/(i-1))*self.costh*self.P1n[i-1, :]-(i/(i-1))*self.P1n[i-2, :]

        # potential 境界条件で使う時間不変量 (毎ステップの再計算を避ける)
        self.sinth = np.sin(grid.th[grid.margin:grid.jxg - grid.margin])
        self.n_values = np.arange(1, self.lmax)
        self.coefficients = (2 * self.n_values + 1) / (2 * self.n_values * (self.n_values + 1))
        self.P1n_reduced = self.P1n[1:, :]
