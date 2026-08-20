"""保存形 (flux form) 離散化の基本演算子.

質量と角運動量を machine precision で保存させるための中核モジュール.

設計方針
--------
保存量 :math:`q` は**セル体積のヤコビアンを吸収した形**で定義する. 球座標
:math:`(r,\\theta)` の軸対称系で, 任意のスカラー密度 :math:`X` に対する発散は

.. math::
   r^2\\sin\\theta\\,\\nabla\\cdot(X\\boldsymbol{v})
     = \\partial_r\\!\\left(r^2\\sin\\theta\\,X v_r\\right)
     + \\partial_\\theta\\!\\left(r\\sin\\theta\\,X v_\\theta\\right)

と書ける (:math:`\\partial_\\theta` の側は :math:`r` が 1 次低いことに注意).
したがって :math:`q = r^2\\sin\\theta X` と置けば

.. math::
   \\frac{\\partial q}{\\partial t} = -\\partial_r F^r - \\partial_\\theta F^\\theta

という**純粋な発散形**になり, 幾何因子由来の源項が一切現れない. これを
セル中心差分で離散化したとき

.. math::
   \\sum_{i,j} q_{ij}\\,\\Delta r\\,\\Delta\\theta

は隣接セルで打ち消し合って (telescoping) 境界フラックスだけが残る.

machine precision 保存に必須の 3 条件
-------------------------------------
1. **フラックスは面で 1 回だけ計算し, 隣接する 2 セルで同一の値を使う.**
   セルごとに再計算すると, 数学的には同じでも丸め誤差が異なるため
   telescoping が崩れる. 本モジュールが面フラックス配列を明示的に持つのは
   このためである.
2. **セル体積で割らない.** :math:`q` がヤコビアンを吸収しているので更新は
   ``q -= dt*(dF^r/dr + dF^θ/dθ)`` のみ. ここで :math:`r^2\\sin\\theta` で
   割ると telescoping が壊れる.
3. **境界面のフラックスはリテラル 0.0 で上書きする.** :math:`\\sin\\theta\\to0`
   や :math:`v\\to0` による「事実上ゼロ」に頼らない. 極では :math:`\\sin\\theta`
   が丸め誤差程度の微小値になり, 保存量が数値的に漏れ出す.

面の添字規約
------------
``S2MFD.Grid`` に合わせ, **面配列の添字 i はセル i-1 と セル i の境界**
(座標 ``grid.RRm[i]``) を指す. 物理セルが ``i in [margin, ixg-margin)`` の
とき, 面は ``i in [margin, ixg-margin]`` が有効で,

* ``RRm[margin]``     : 内側境界 (``rrmin``)
* ``RRm[ixg-margin]`` : 外側境界 (``rrmax``)

となる (``grid.RRm[margin] == grid.rrmin`` はビット一致することを
``tests/test_conservation.py`` で検証している). セル i の発散は
``(F[i+1] - F[i])/drr`` で得られる.

参考
----
齋藤 (2024) 修士論文の平均場モデルコード (``mhd.f90`` / ``calculation.f90``)
の定式化を Python/numba に移植したもの. 保存量に :math:`\\Omega_1` のみを
入れ, フラックスには :math:`\\Omega_0+\\Omega_1` を運ばせる点が数値的な鍵で,
:math:`\\Omega_0` 込みの「素朴な保存形」では剛体回転からの出発時に
:math:`\\Omega_1` の増分が丸めで消える (``doc/dev_records`` 参照).
"""
import numpy as np
from numba import njit

from ._jit import kernel, prange

__all__ = [
    'face_average_r', 'face_average_th',
    'zero_boundary_faces_r', 'zero_boundary_faces_th',
    'flux_divergence', 'add_flux_divergence', 'add_flux_divergence_scaled',
    'add_flux_work',
    'cell_integral',
]


@kernel()
def face_average_r(qq, margin, out):
    """セル中心量から r 面の値を算術平均で作る.

    ``out[i] = 0.5*(qq[i-1] + qq[i])`` を ``i in [margin, ixg-margin]``
    に対して計算する. 境界面 (``i = margin`` と ``i = ixg-margin``) も
    埋めるが, 保存が必要なフラックスでは呼び出し側で
    :func:`zero_boundary_faces_r` によって 0.0 に上書きされる.

    Parameters
    ----------
    qq : numpy.ndarray
        セル中心量 (ixg, jxg).
    margin : int
        ゴーストセル数.
    out : numpy.ndarray
        面配列 (ixg, jxg). 全域が上書きされるわけではないので,
        呼び出し側であらかじめゼロ初期化しておくこと.
    """
    ixg, jxg = qq.shape
    for i in prange(margin, ixg - margin + 1):
        for j in range(jxg):
            out[i, j] = 0.5 * (qq[i - 1, j] + qq[i, j])


@kernel()
def face_average_th(qq, margin, out):
    """セル中心量から theta 面の値を算術平均で作る.

    ``out[j] = 0.5*(qq[j-1] + qq[j])`` を ``j in [margin, jxg-margin]``
    に対して計算する.
    """
    ixg, jxg = qq.shape
    for i in prange(ixg):
        for j in range(margin, jxg - margin + 1):
            out[i, j] = 0.5 * (qq[i, j - 1] + qq[i, j])


@njit(fastmath=False)
def zero_boundary_faces_r(ff, margin):
    """r 方向の物理境界面のフラックスをリテラル 0.0 にする.

    これにより保存量は r 境界から一切出入りしなくなる. 「速度がゼロだから
    フラックスもゼロ」ではなく**代入によって厳密にゼロ**にする点が重要で,
    人工粘性や Maxwell 応力のように速度に比例しないフラックスでも
    同じ規律を適用する.
    """
    ixg, jxg = ff.shape
    for j in range(jxg):
        ff[margin, j] = 0.0
        ff[ixg - margin, j] = 0.0


@njit(fastmath=False)
def zero_boundary_faces_th(ff, margin):
    """theta 方向の物理境界面 (極) のフラックスをリテラル 0.0 にする.

    極では ``sin(theta) -> 0`` なので幾何因子から自動的にゼロになりそうに
    見えるが, 実際には ``THm[margin]`` が厳密に 0 でも下流の演算で丸め誤差が
    入りうる. 代入で潰しておく.
    """
    ixg, jxg = ff.shape
    for i in range(ixg):
        ff[i, margin] = 0.0
        ff[i, jxg - margin] = 0.0


@kernel()
def add_flux_divergence(dqq, ffr, ffth, drr, dth, margin):
    """フラックスの発散を ``dqq`` から差し引く (``dqq -= div F``).

    ``dqq[i,j] -= (ffr[i+1,j]-ffr[i,j])/drr + (ffth[i,j+1]-ffth[i,j])/dth``

    セル体積で割らないのが要点. 保存量 ``q`` はヤコビアンを吸収済みなので,
    ``sum(q)*drr*dth`` がそのまま体積積分になり telescoping する.

    Parameters
    ----------
    dqq : numpy.ndarray
        時間微分の累積先 (ixg, jxg). その場で更新される.
    ffr, ffth : numpy.ndarray
        r 面 / theta 面のフラックス配列 (ixg, jxg).
    drr, dth : float
        格子間隔.
    margin : int
        ゴーストセル数.
    """
    ixg, jxg = dqq.shape
    idrr = 1.0 / drr
    idth = 1.0 / dth
    for i in prange(margin, ixg - margin):
        for j in range(margin, jxg - margin):
            dqq[i, j] -= ((ffr[i + 1, j] - ffr[i, j]) * idrr
                          + (ffth[i, j + 1] - ffth[i, j]) * idth)


@kernel()
def add_flux_divergence_scaled(dqq, ffr, ffth, drr, dth, margin, scale):
    """発散に動径依存の係数を掛けて差し引く (``dqq -= scale(r) * div F``).

    音速抑制法の :math:`\\xi_s^{-2}` のように, 発散全体に動径方向の係数が
    掛かる場合に使う. このとき保存するのは :math:`\\sum q` ではなく
    :math:`\\sum q/{\\rm scale}` で, ``scale`` が定数でない場合でも
    telescoping はその重み付き和について成り立つ.

    Parameters
    ----------
    scale : numpy.ndarray
        動径方向の係数 (``ixg``,).
    """
    ixg, jxg = dqq.shape
    idrr = 1.0/drr
    idth = 1.0/dth
    for i in prange(margin, ixg - margin):
        sc = scale[i]
        for j in range(margin, jxg - margin):
            dqq[i, j] -= sc*((ffr[i + 1, j] - ffr[i, j])*idrr
                             + (ffth[i, j + 1] - ffth[i, j])*idth)


@njit(fastmath=False)
def flux_divergence(ffr, ffth, drr, dth, margin):
    """フラックスの発散を新しい配列として返す (``-div F``).

    :func:`add_flux_divergence` の非破壊版. 診断用.
    """
    dqq = np.zeros_like(ffr)
    add_flux_divergence(dqq, ffr, ffth, drr, dth, margin)
    return dqq


@kernel()
def add_flux_work(heat, ffr, ffth, uu, drr, dth, margin):
    """局所的なエネルギー変換率 :math:`-F\\cdot\\nabla u` を ``heat`` に加算する.

    保存量の更新が :math:`\\partial q/\\partial t = -\\nabla\\cdot F` の形の
    とき, 対応するエネルギーの変化率は

    .. math::
       u\\frac{\\partial q}{\\partial t}
         = -\\nabla\\cdot(uF) + F\\cdot\\nabla u

    と分解できる. 第 1 項は輸送 (体積積分すると境界だけに残る), 第 2 項が
    **局所的な**変換である. したがって散逸加熱は :math:`-F\\cdot\\nabla u`.

    なぜ :math:`u\\,\\partial q/\\partial t` をそのまま使ってはいけないか
    -----------------------------------------------------------------------
    体積積分すれば両者は一致するが, **局所的には発散の分だけずれる**.
    角運動量の場合 :math:`u=\\Omega_0+\\Omega_1` で
    :math:`\\Omega_0\\gg\\Omega_1` なので, このずれは正しい値の
    :math:`\\Omega_0/\\Omega_1` 倍 (20-100 倍) にもなる. しかも符号が
    空間的に振動するため, エントロピー方程式に偽の双極子を作り, 浮力を
    通じて計算を不安定にする. 実際にこれが原因で発散していた
    (``doc/dev_records/2026-08-20_rempel2006_worklog.md``).

    この形なら :math:`\\Omega_0` は微分で消えるので現れない.

    離散化と正値性
    --------------
    面フラックスと面での勾配を組にして, セル中心へ平均する:

    .. math::
       Q_{ij} = -\\tfrac12\\left[
         F^r_i\\frac{u_i-u_{i-1}}{\\Delta r}
       + F^r_{i+1}\\frac{u_{i+1}-u_i}{\\Delta r}\\right] - (\\theta 方向)

    拡散フラックス :math:`F=-\\kappa J\\Delta u/\\Delta r` に対しては
    各項が :math:`+\\kappa J(\\Delta u/\\Delta r)^2\\ge0` になるので,
    **散逸の正値性が離散レベルで保証される**.
    """
    ixg, jxg = heat.shape
    idrr = 1.0/drr
    idth = 1.0/dth
    for i in prange(margin, ixg - margin):
        for j in range(margin, jxg - margin):
            heat[i, j] -= 0.5*(
                ffr[i, j]*(uu[i, j] - uu[i - 1, j])*idrr
                + ffr[i + 1, j]*(uu[i + 1, j] - uu[i, j])*idrr
                + ffth[i, j]*(uu[i, j] - uu[i, j - 1])*idth
                + ffth[i, j + 1]*(uu[i, j + 1] - uu[i, j])*idth)


@njit(fastmath=False)
def cell_integral(qq, drr, dth, margin):
    """保存量の体積積分 ``sum(q)*drr*dth`` を物理セルについて計算する.

    ``q`` はヤコビアン ``r^2 sin(theta)`` を吸収済みなので, これがそのまま
    (方位角方向の :math:`2\\pi` を除いた) 体積積分になる. 保存則の検証に
    使う.

    総和は Neumaier 補正付きで取り, 積分そのものの丸め誤差が
    離散化由来の保存誤差を覆い隠さないようにしている.
    """
    ixg, jxg = qq.shape
    total = 0.0
    comp = 0.0
    for i in range(margin, ixg - margin):
        for j in range(margin, jxg - margin):
            x = qq[i, j]
            t = total + x
            if abs(total) >= abs(x):
                comp += (total - t) + x
            else:
                comp += (x - t) + total
            total = t
    return (total + comp) * drr * dth
