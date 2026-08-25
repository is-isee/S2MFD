"""カーネルの JIT デコレータと並列化スイッチ。

環境変数 ``S2MFD_PARALLEL`` でスレッド並列を切り替える。

* ``S2MFD_PARALLEL=0`` (既定) — ``njit`` のみ。逐次。
* ``S2MFD_PARALLEL=<n>`` — ``njit(parallel=True)`` で ``prange`` を
  ``n`` スレッドに展開する (``n`` が数値ならスレッド数としても使う)。

なぜ既定を逐次にしてあるか
--------------------------
numba の並列領域には 1 回あたり数 10 us のスレッド起動コストがある。
1 ステップで 15-20 回カーネルを呼ぶので、格子が小さいとこの固定費だけで
逐次より遅くなる。実測 (Xeon Gold 6326, slope-limited diffusion の
動径フラックス) では

===========  =========  ==================================
格子          逐次       並列の最良 (スレッド数)
===========  =========  ==================================
64 x 64       39 us     35 us  (4)   — ほぼ効果なし
108 x 144    145 us     59 us  (4)   — 2.45 倍
258 x 290    946 us    158 us  (64)  — 5.98 倍
===========  =========  ==================================

つまり **効くのは 100^2 格子以上、かつスレッドは 3-4 で頭打ち**。
64^2 では 12 スレッド以上でむしろ遅くなる。小さい格子を多数流すとき
(パラメタスキャンなど) は逐次のままプロセス並列にするほうが速い。

並列化してよいカーネルの条件
----------------------------
``prange`` にするのは **反復間に依存のない、要素ごとの書き込みだけ**の
ループに限る。これを守れば結果はスレッド数によらずビット単位で同一に
なる (浮動小数点の加算順序が変わらないため)。

``cell_integral`` のような **総和はここに含めない**。Neumaier 補正加算の
順序が変わると保存量の機械精度が壊れるので、意図的に逐次のままにする。
"""
import os

from numba import njit, prange  # noqa: F401  (prange は各モジュールから使う)

_env = os.environ.get('S2MFD_PARALLEL', '0')

#: スレッド並列が有効か。
PARALLEL = _env not in ('0', '', 'false', 'False', 'no')

if PARALLEL:
    try:
        _nthreads = int(_env)
    except ValueError:
        _nthreads = 0
    if _nthreads > 0:
        import numba
        numba.set_num_threads(_nthreads)


def kernel(**kwargs):
    """並列化して安全なカーネル用の ``njit`` デコレータ。

    ``S2MFD_PARALLEL`` が有効なら ``parallel=True`` を付ける。無効なら
    ``prange`` は numba によって ``range`` に落ちるので、コードは 1 つの
    まま両方の実行形態を取れる。
    """
    return njit(fastmath=False, parallel=PARALLEL, **kwargs)
