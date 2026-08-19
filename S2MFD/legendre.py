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
        self.build_potential_operator(grid)

    def build_potential_operator(self, grid):
        """potential 境界条件の「表面値 → ゴースト値」線形作用素を作る。

        境界条件は Aφ について線形なので、ルジャンドル射影と外部ポテンシャル場
        による再構成をまとめて1つの行列で表せる::

            Aph_ghost[k][j'] = sum_j Aph_surface[j] * M[k][j, j']

        ここで
        M[k][j,j'] = sin(th_j) dth * sum_n c_n (r_top/r_ghost_k)^(n+1)
                                          P1n(th_j) P1n(th_j')

        毎ステップ生成していた (lmax-1, jx) の一時配列と冪計算が、
        1 回の行列ベクトル積 (jx x jx) に置き換わる。
        """
        margin = grid.margin
        r_top = grid.rr[grid.ixg - margin - 1]
        L = self.P1n_reduced                       # (lmax-1, jx)
        row_w = self.sinth * grid.dth              # (jx,)
        self.potential_operator = np.empty((margin, grid.jx, grid.jx))
        for k in range(margin):
            r_ghost = grid.rr[grid.ixg - k - 1]
            scale = self.coefficients * (r_top / r_ghost)**(self.n_values + 1)
            self.potential_operator[k] = row_w[:, None] * (L.T @ (scale[:, None] * L))
