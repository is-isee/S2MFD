コードを読むための Python
==============================================

この章は **S2MFD のコードを読むときに出てくる Python の書き方**\ を、
出てくる順にひとつずつ説明する。物理や数値計算の中身は
:doc:`dynamic` に、使い方は :doc:`usage` にある。

.. note::
   **前提にしている知識は次だけ。**

   * 変数、``if``\ 、``for``
   * ``def`` で関数を作って ``return`` で返す
   * リストと辞書 (``[1,2,3]``\ 、``{'a':1}``\ )
   * ``import numpy as np`` して配列を作り、足したり掛けたりする

   ``class`` は知らないものとして書いてある。デコレータ (``@`` の付いた
   行)、クロージャ、``**kwargs`` なども全部この章で説明する。

.. contents:: 内容
   :local:
   :depth: 2


0. 先に地図を持つ
--------------------------------------------------

本体 ``S2MFD/`` は 8,000 行ほどで、思っているより小さい。

.. list-table::
   :header-rows: 1
   :widths: 34 12 54

   * - ファイル
     - 行数
     - 中身
   * - ``physics/hydro.py``
     - 889
     - 流体 (密度・エントロピー・角運動量・子午面流) の右辺
   * - ``physics/dynamic.py``
     - 851
     - 動力学モードの時間積分を束ねる ``DynamicSolver``
   * - ``physics/physics_core.py``
     - 766
     - 磁場の誘導方程式。**計算の心臓部**
   * - ``physics/artdif.py``
     - 560
     - 人工拡散 (slope-limited diffusion)
   * - ``simulation.py``
     - 378
     - 運動学モードの実行ループ
   * - ``setup.py`` / ``stratification.py`` / ``grid.py`` / ``cfg.py``
     - 235-333
     - 背景場・成層・格子・設定
   * - その他
     - < 320
     - 境界条件、エネルギー収支、安定性、遺伝的アルゴリズムなど

読む順としては ``cfg.py`` → ``grid.py`` → ``setup.py`` →
``physics/physics_core.py`` が素直。前の 3 つは「準備」、最後が「計算」。

そして、**言語機能としては驚くほど何も使っていない**\ 。本体 8,000 行の中に

* メタクラス、``__getattr__``\ 、``__slots__`` … **0 個**
* ジェネレータ (``yield``\ ) … **0 個**
* ``functools``\ 、型注釈 (``typing``\ )、``async`` … **0 個**
* クラスの継承 (``super()``\ ) … **1 か所だけ**

つまり、この章に書いてあることさえ分かれば、他に隠し球はない。


1. クラス — 「まとめて持ち歩くための箱」
--------------------------------------------------

1.1 なぜ必要か
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

数値計算では、ほとんどの関数が同じものを欲しがる。半径の配列、角度の
配列、格子間隔、粘性係数、境界の位置……。素直に関数の引数にすると
こうなる。実際 ``physics_core.py`` の一番内側の計算関数は

.. code-block:: python

    def kernel(Bph, Aph, dt, rr, sth, rrm, sthm, drr, drrm, drr2_, dth,
               urr, uth, et, etrr, omrr, omth, so, ibase, alpha_code,
               inv_rr, inv_rr2, inv_sth, inv_sth2, alpha_fac, Bphm, Aphm,
               urr_u, urr_v, uth_u, uth_v, et_u, et_v, etrr_u, etrr_v,
               omth_u, omth_v, so_u, so_v):

**引数が 39 個ある**\ 。これを人間が毎回書くのは無理だし、順番を 1 つ
間違えれば静かに間違った答えが出る (実際に起きた。10 節を見よ)。

そこで、関係するものをひとかたまりにして名前を付ける。それがクラス。

.. code-block:: python

    def time_marching(Bph, Aph, dt, cfg, grid, setup):   # 引数 6 個

``cfg`` は設定一式、``grid`` は格子一式、``setup`` は背景場一式。
39 個が 3 個に減っている。``time_marching`` の中で ``grid.drr``\ 、
``grid.dth`` のように取り出して、39 個の引数に展開してから
``kernel`` を呼ぶ。**クラスは「引数の束」だと思ってよい。**

1.2 一番小さいクラス
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    class Box:
        def __init__(self, a, b):
            self.a = a
            self.b = b

        def total(self):
            return self.a + self.b

使い方:

.. code-block:: python

    >>> x = Box(3, 4)      # ここで __init__ が呼ばれる
    >>> x.a
    3
    >>> x.total()
    7

読み方はこうだ。

``class Box:``
    「``Box`` という種類の箱を定義する」という宣言。この行だけでは
    まだ箱は 1 つも作られていない。**設計図を書いただけ。**

``x = Box(3, 4)``
    設計図から箱を 1 つ作る。作られた箱を **インスタンス**\ という。
    ``Box(3, 4)`` と書くと Python が自動的に ``__init__`` を呼ぶ。

``self``
    **「いま作られている、あるいはいま操作されている箱そのもの」**\ 。
    ここが最初の関門なので、次で丁寧にやる。

1.3 ``self`` の正体
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``self`` は魔法ではなく、**ただの第 1 引数**\ である。次の 2 行は
Python の内部ではまったく同じ意味になる。

.. code-block:: python

    x.total()          # ふだんこう書く
    Box.total(x)       # 実際に起きているのはこれ

つまり ``x.total()`` と書くと、Python が ``x`` を勝手に第 1 引数として
渡している。だから ``def total(self):`` の ``self`` には ``x`` が入る。

``self.a = a`` の意味も同じで、「この箱の ``a`` という引き出しに、
引数の ``a`` を入れる」。左の ``self.a`` と右の ``a`` は別物である。

.. tip::
   ``self`` という名前に意味はない。``def total(hako):`` と書いて
   ``hako.a`` としても動く。全員が ``self`` と書く約束になっているだけ。

1.4 ふつうの関数との違いは「状態が残る」こと
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

関数は呼ぶたびに何も覚えていない。クラスは覚えている。

.. code-block:: python

    >>> g1 = S2MFD.Grid.from_cfg(cfg1)     # 108x72 の格子
    >>> g2 = S2MFD.Grid.from_cfg(cfg2)     # 216x144 の格子
    >>> g1.ix, g2.ix
    (108, 216)

``g1`` と ``g2`` は同じ設計図から作られた別の箱で、中身は独立している。
解像度を振る実験ができるのはこのおかげ。

1.5 実物を見る
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

対話的に触るのが一番早い。

.. code-block:: python

    >>> import S2MFD
    >>> cfg = S2MFD.build_cfg('parameters/rempel06_paper.py')
    >>> grid = S2MFD.Grid.from_cfg(cfg)
    >>> grid.rr.shape          # 半径の配列の形
    (112,)
    >>> grid.RR.shape          # 半径の 2 次元配列
    (112, 76)
    >>> [k for k in vars(grid)][:6]    # 箱の引き出しを全部見る
    ['ix', 'jx', 'margin', 'rrmin', 'rrmax', 'thmin']

``vars(obj)`` で中身が全部見られる。**知らないクラスに出会ったら、
まず ``vars()`` を打つ。**


2. クラスに付く 4 つの飾り
--------------------------------------------------

S2MFD のクラスに出てくる ``@`` 付きの行は 4 種類しかない。

2.1 ``@dataclass`` — ``__init__`` を書かなくてよくする
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

引き出しの名前を並べるだけのクラスで、``__init__`` に
``self.a = a`` を延々と書くのは退屈である。``@dataclass`` を付けると
Python がその ``__init__`` を自動生成してくれる。

.. code-block:: python

    from dataclasses import dataclass

    @dataclass
    class GAConfig:
        threshold: float = 0.80
        max_generations: int = 50
        mutation_probability: float = 0.3

これは「``__init__`` の中で ``self.threshold = threshold`` などを
書いたクラス」と同じ意味になる。既定値も付けられる。

.. code-block:: python

    >>> c = GAConfig()                  # 全部既定値
    >>> c = GAConfig(max_generations=100)   # 一部だけ変える
    >>> c.threshold
    0.8

``float`` や ``int`` の部分は **型注釈**\ という。Python はこれを
チェックしない (``threshold`` に文字列を入れても動いてしまう)。
人間へのメモであり、``@dataclass`` にとっては「これは引き出しですよ」
という目印でもある。

``grid.py`` の ``Grid`` にはもう一段だけ細工がある。

.. code-block:: python

    ixg: int = field(init=False)

``field(init=False)`` は「これは引き出しだが、**外から渡すのではなく
中で計算する**」という指定。``ixg = ix + 2*margin`` なので、外から
渡させたら矛盾しうる。矛盾させない仕組みである。

2.2 ``@classmethod`` — もう一つの作り方
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``Grid`` は引数を 9 個取るが、ふつうは設定ファイルから全部決まる。
そこで「``cfg`` から作る」という専用の入口がある。

.. code-block:: python

    @classmethod
    def from_cfg(cls, cfg):
        return cls(
            ix=cfg.ix, jx=cfg.jx, margin=cfg.margin,
            rrmin=cfg.rrmin, rrmax=cfg.rrmax,
            ...
        )

読み方:

* ``@classmethod`` が付くと、第 1 引数が ``self`` (箱) ではなく
  ``cls`` (**設計図そのもの**\ ) になる。ここでは ``cls`` は ``Grid``\ 。
* だから ``cls(...)`` は ``Grid(...)`` と書いたのと同じで、
  ふつうの作り方を呼んでいる。
* 使うときは箱を作る前に呼ぶ: ``S2MFD.Grid.from_cfg(cfg)``\ 。

なぜこうするか。**格子の作り方を 1 か所に決めるため**\ である。
``getattr(cfg, 'grid_stretch', 0.0)`` のような既定値の解釈がここに
しかないので、実行スクリプトごとに違う格子ができる事故が起きない。

2.3 ``@property`` — 括弧のいらないメソッド
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    @property
    def _top_is_pole(self):
        """theta 方向の上端が極か (True) 赤道か (False)."""
        thmax = getattr(self.cfg, 'thmax', np.pi)
        return abs(thmax - 0.5*np.pi) > 1.0e-9

``@property`` が付くと、**呼び出しの括弧が要らなくなる**\ 。

.. code-block:: python

    >>> sol._top_is_pole        # sol._top_is_pole() ではない
    True

見た目は引き出し (``grid.ix`` のような) だが、中身は毎回計算される。
「``cfg`` から決まるので保存しておくと食い違う」量に使う。

2.4 継承とミックスイン — ``NpzIO``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``class Grid(NpzIO):`` の括弧の中が **親クラス**\ 。「``NpzIO`` が
持っている機能を ``Grid`` も持つ」という意味である。S2MFD では
``NpzIO`` を通じて ``save`` / ``load`` を配っているだけで、
**深い継承は一切していない** (``super()`` の呼び出しは全体で 1 か所)。

``npz_io.py`` の全文は 40 行しかないので、そのまま読める。

.. code-block:: python

    class NpzIO:
        def save(self, filename):
            np.savez(filename, **self.__dict__)

        @classmethod
        def load(cls, filename):
            data = np.load(filename)
            obj = cls.__new__(cls)
            obj.__dict__.update({
                key: (data[key].item() if data[key].ndim == 0 else data[key])
                for key in data.files
            })
            return obj

ここに 3 つ新しい道具が出てくる。

``self.__dict__``
    **箱の引き出しは、実は辞書である。** ``grid.ix`` は
    ``grid.__dict__['ix']`` と同じものを指す。``vars(grid)`` はこの
    辞書を返しているだけ。だから ``np.savez(filename, **self.__dict__)``
    は「引き出しを全部ファイルに書く」になる。

``**`` (辞書のばらし)
    ``f(**{'a':1, 'b':2})`` は ``f(a=1, b=2)`` と同じ。辞書のキーを
    キーワード引数の名前として展開する。6 節で改めて扱う。

``cls.__new__(cls)``
    **``__init__`` を通さずに空の箱を作る。** ``load`` では中身を
    ファイルから入れるので、``__init__`` の計算 (格子を作り直すなど) を
    もう一度やる必要がないし、やると保存した値と食い違いかねない。
    これは S2MFD の中で一番「裏技」に近い 1 行だが、ここ以外には
    出てこない。


3. デコレータの正体 (``@`` の付いた行)
--------------------------------------------------

3.1 ``@`` は関数の書き換え
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

デコレータには特別な文法上の意味はない。次の 2 つは完全に同じである。

.. code-block:: python

    @njit
    def advection(Bph, Aph, ...):
        ...

.. code-block:: python

    def advection(Bph, Aph, ...):
        ...
    advection = njit(advection)

つまり ``@なにか`` は「**定義した関数をその「なにか」に一度通して、
返ってきたものを同じ名前に付け直す**」という省略記法にすぎない。

``njit`` は関数を受け取って「機械語にコンパイルする版の関数」を返す。
``property`` は関数を受け取って「括弧なしで読める版」を返す。
``dataclass`` はクラスを受け取って「``__init__`` を足したクラス」を返す。
やっていることは全部同じ形である。

3.2 デコレータを **返す** 関数
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``@njit(fastmath=False)`` のように括弧が付くと一段深くなる。
``njit(fastmath=False)`` がまずデコレータを作り、それが関数に適用される。

.. code-block:: python

    @njit(fastmath=False)
    def f(x): ...

    # 同じ意味
    def f(x): ...
    deco = njit(fastmath=False)   # ← デコレータを作る
    f = deco(f)                   # ← それを適用する

S2MFD ではこれを自前でも作っている。``S2MFD/physics/_jit.py``\ :

.. code-block:: python

    PARALLEL = os.environ.get('S2MFD_PARALLEL', '0') not in ('0', '', 'false')

    def kernel(**kwargs):
        """並列化して安全なカーネル用の njit デコレータ。"""
        return njit(fastmath=False, parallel=PARALLEL, **kwargs)

``kernel`` は関数ではなく **デコレータを返す関数**\ だから、使うときは
必ず括弧が要る。

.. code-block:: python

    @kernel()
    def slope_limited_flux(...):
        ...

これで何が嬉しいか。**環境変数 1 つで、全カーネルの並列化を一斉に
切り替えられる。**

.. code-block:: bash

    S2MFD_PARALLEL=0 python run.py     # 逐次
    S2MFD_PARALLEL=4 python run.py     # 4 スレッド

``@njit(parallel=True)`` と 21 か所 (``@kernel()`` を使っているカーネルの数)
に直書きしてあったら、こうはできない。


4. 関数を作る関数 (クロージャ)
--------------------------------------------------

ここが本体で一番込み入った部分だが、**込み入っている理由は性能だけ**\ で
ある。``physics_core.py:207``\ :

.. code-block:: python

    def _make_time_marching_kernel(fast, nonlocal_alpha, separable=False):
        """time_marching の内部カーネルを生成する。"""

        @njit(fastmath=fast, boundscheck=False)
        def kernel(Bph, Aph, dt, ...):
            ...
            if nonlocal_alpha:          # ← 外側の引数をここで使っている
                src = so[i, j]*alpha_fac[j]
            else:
                src = so[i, j]*Bph[i, j]/(1.0 + Bph[i, j]**2)
            ...

        return kernel

**関数の中で関数を定義して、それを返している。** 内側の ``kernel`` は
外側の引数 ``fast`` / ``nonlocal_alpha`` を覚えたまま外に出ていく。
このような「外側の変数を抱えた関数」を **クロージャ** という。

なぜこうするか。``nonlocal_alpha`` による分岐を **内側の二重ループの中に
残したくない**\ からである。10 万セルを毎ステップ回るループの中に
``if`` があると、CPU のベクトル化が効かず 2 倍以上遅くなる。関数を作る
時点で分岐を決めてしまえば、numba がコンパイルする時に ``if`` ごと
消える。

.. note::
   **クロージャは「引数を先に一部だけ埋めた関数」だと思うとよい。**

   .. code-block:: python

       def kakezan(a):
           def f(x):
               return a*x
           return f

       nibai = kakezan(2)      # a=2 を埋めた関数ができる
       nibai(10)               # -> 20

   ``_make_time_marching_kernel(True, False, False)`` は、
   ``fast=True, nonlocal_alpha=False, separable=False`` を埋めた
   計算カーネルを作っている。それだけのこと。

4.1 作ったものを取っておく (辞書によるキャッシュ)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

numba のコンパイルは数秒かかるので、毎回作り直すと話にならない。
一度作ったら辞書に入れておく。

.. code-block:: python

    _TIME_MARCHING_KERNELS = {}

    def get_time_marching_kernel(fast, nonlocal_alpha, separable):
        key = (bool(fast), bool(nonlocal_alpha), bool(separable))
        fn = _TIME_MARCHING_KERNELS.get(key)
        if fn is None:                              # まだ作っていない
            fn = _make_time_marching_kernel(*key)
            _TIME_MARCHING_KERNELS[key] = fn
        return fn

新しい書き方が 3 つ。

``key = (a, b, c)``
    **タプル**\ 。リストと違って中身を変えられない。変えられないものは
    辞書のキーにできる (リストはできない)。「3 つの真偽値の組」を
    1 つの名前として使っている。

``辞書.get(key)``
    ``辞書[key]`` はキーが無いとエラーになるが、``.get(key)`` は
    ``None`` を返す。「あれば使う、なければ作る」の定型句。

``f(*key)``
    ``*`` はタプルやリストを **位置引数にばらす**\ 。
    ``f(*(1,2,3))`` は ``f(1,2,3)``\ 。``**`` が辞書をキーワード引数に
    ばらすのに対し、``*`` は並びを位置引数にばらす。

先頭のアンダースコア (``_TIME_MARCHING_KERNELS``\ 、
``_make_time_marching_kernel``\ ) は「**このファイルの中だけの都合**\ 。
外から使わないでほしい」という慣習的な目印。Python は強制しない。


5. 辞書と ``lambda`` を「表」として使う
--------------------------------------------------

5.1 分岐の代わりの辞書
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    ALPHA_CODE = {'BL': 0, 'H10': 0, 'normal': 1, 'R06': 0}

これは ``if alpha_type == 'BL': code = 0 elif ...`` と同じことを表で
書いたもの。**新しい α 効果を足すときに 1 行足すだけで済む**\ ので、
分岐が散らばらない。

5.2 ``lambda`` = 名前のない小さな関数
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    f = lambda x, y: x + y      # これは
    def f(x, y): return x + y   # これと同じ

``def`` と違って **式を 1 つ書くだけ**\ で、``return`` も書かない。
表の中に式を埋め込みたいときに使う。``cfg.py:8``\ :

.. code-block:: python

    _DERIVATIONS = [
        ('ome', ('com', 'RSUN', 'ett'), lambda com, RSUN, ett: com/RSUN**2*ett),
        ('omc', ('ome',),               lambda ome: 0.92*ome),
        ('c2',  ('ome',),               lambda ome: 0.2*ome),
        ...
    ]

読み方は「``ome`` という量は ``com``\ 、``RSUN``\ 、``ett`` から
``com/RSUN**2*ett`` で決まる」。これが 8 行並んでいる。

この表があるおかげで、``cfg.rey = 1400`` と基本量を書き換えたときに
``cfg.resolve()`` が派生量 ``uu0`` を計算し直せる。式が
``defaults.py`` の中に散らばっていたら、こうはいかない。

.. code-block:: python

    for name, inputs, fn in _DERIVATIONS:        # 表を 1 行ずつ
        value = fn(*[getattr(self, i) for i in inputs])
        setattr(self, name, value)

``[getattr(self, i) for i in inputs]``
    **リスト内包表記**\ 。``inputs`` の各名前について ``getattr`` した
    結果を並べたリストを作る。``for`` で 3 行書くのと同じ。

``fn(*[...])``
    できたリストを ``*`` で位置引数にばらして ``lambda`` に渡す。

``getattr`` / ``setattr``
    次節。


6. 引数と属性の小道具
--------------------------------------------------

6.1 キーワード引数と既定値
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def analyse(tag, steady_yr=20.0):
        ...

    analyse('v2_a125')                  # steady_yr は 20.0
    analyse('v2_a125', steady_yr=40.0)  # 明示すると上書き

引数が多い関数では **名前で渡す**\ 。順番を覚えなくてよくなり、
読む人にも意味が分かる。S2MFD では引数の取り違えで実際にバグを出した
ことがあるので (10 節)、**新しい呼び出しは名前付きで書く**\ 決まりに
している。

6.2 ``**kwargs`` — 残り全部を辞書で受け取る
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def kernel(**kwargs):
        return njit(fastmath=False, parallel=PARALLEL, **kwargs)

``kernel(cache=True)`` と呼ぶと、関数の中では
``kwargs = {'cache': True}`` という辞書になる。そして
``njit(..., **kwargs)`` でまたばらして渡す。「**自分は知らない引数を
そのまま下に流す**」ときの書き方。

まとめると:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - 書き方
     - 意味
   * - ``f(*lst)``
     - リスト・タプルを位置引数にばらす
   * - ``f(**dic)``
     - 辞書をキーワード引数にばらす
   * - ``def f(*args)``
     - 余った位置引数をタプルで受ける (S2MFD では未使用)
   * - ``def f(**kwargs)``
     - 余ったキーワード引数を辞書で受ける

6.3 ``getattr(obj, '名前', 既定値)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``obj.名前`` は名前が無いとエラーになる。あるかどうか分からないときは

.. code-block:: python

    stretch = getattr(cfg, 'grid_stretch', 0.0)

と書く。「``cfg`` に ``grid_stretch`` があればその値、無ければ ``0.0``」。
本体には ``getattr`` が 67 か所あるが、**全部この 1 つの形**\ である。

なぜ要るか。パラメタファイルは古いものも新しいものも読めなければ
ならないから。新しい機能 (伸縮格子) を足したとき、既存のパラメタ
ファイルに ``grid_stretch`` を書き足して回らずに済む。

``setattr(obj, '名前', 値)`` はその逆で、名前を **文字列で指定して**
代入する。5.2 の派生量の表のように、名前が実行時に決まるときに使う。


7. モジュール、``import``\ 、設定ファイルの読み込み
------------------------------------------------------------

7.1 ``import`` のかたち
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import numpy as np                    # np.array(...)
    from S2MFD.npz_io import NpzIO        # NpzIO をこの名前で使う
    from S2MFD.parameters.defaults import *   # 中身を全部持ってくる

3 つ目の ``import *`` はふつう避けるべき書き方だが、
**パラメタファイルではわざと使っている**\ 。

.. code-block:: python

    # parameters/rempel06_paper.py
    from S2MFD.parameters.rempel06 import *
    rrmax = 0.985*RSUN     # 差分だけ書く

「既定値を全部引き継いで、変えたいものだけ書く」が意図。

.. warning::
   その代償として、**パラメタファイルを ``grep`` しても値が出てこない
   ことがある**\ 。継承元から黙って入ってくるので、確かめたいときは
   ファイルを読むのではなく実際の値を出す。

   .. code-block:: bash

       $ grep boundary_condition_type S2MFD/parameters/rempel06_paper.py
       $                                      # ← 1 行も出てこない

   .. code-block:: python

       >>> cfg = S2MFD.build_cfg('parameters/rempel06_paper.py')
       >>> cfg.rrmax/cfg.RSUN, cfg.boundary_condition_type
       (0.985, 'vertical')

   値は継承元の ``rempel06.py`` から入っている。これは実際に見落としの
   原因になった (磁場の下部境界条件が論文の指定と違っていた)。
   ``doc/dev_records/2026-08-23_paper_audit2.md`` 参照。

7.2 ``importlib`` — ``.py`` を設定ファイルとして読む
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

S2MFD の設定は YAML でも JSON でもなく **Python ファイル**\ である。
``S2MFD/cfg.py``\ :

.. code-block:: python

    spec = importlib.util.spec_from_file_location("parameters", parameter_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)              # ← ここで実行される

    for k, v in vars(module).items():
        if not k.startswith("__"):
            setattr(self, k, v)

やっていることは 3 段階。

1. 指定されたパスの ``.py`` を **モジュールとして読み込んで実行する**
2. ``vars(module)`` でそのファイルが定義した名前を全部取り出す
   (1.5 節の ``vars`` と同じもの。モジュールの引き出しも辞書である)
3. ``__`` で始まる Python 内部用のものを除いて、``setattr`` で
   ``cfg`` の引き出しに移す

つまり ``cfg.rrmax`` は、パラメタファイルに書いた ``rrmax = ...`` が
そのまま入っている。設定ファイルが Python なので
``rrmax = 0.985*RSUN`` のように **式が書ける**\ のが利点である。


8. numpy の約束事
--------------------------------------------------

言語機能ではないが、実際に間違えるのはここ。

8.1 スライスは **コピーではない**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    >>> a = np.arange(5)
    >>> b = a[1:4]
    >>> b[0] = 100
    >>> a
    array([  0, 100,   2,   3,   4])

リストのスライスはコピーだが、**numpy 配列のスライスは元の配列への
「窓」(view) である**\ 。書き換えると元も変わる。

これは欠点ではなく、この種のコードでは必須の性質である。境界条件を

.. code-block:: python

    Bph[:, 0] = 0.0        # 元の配列が直接書き換わる

と書けるのはこのおかげ。逆に、値を取っておきたいときは
``a[1:4].copy()`` と明示する必要がある。

8.2 ゴーストセルと ``margin``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

差分法では隣のセルが要るので、計算領域の外側に **ゴーストセル**\ を
``margin`` 枚ぶん持つ。

.. code-block:: text

    |<-- margin -->|<------ 物理セル ix 個 ------>|<-- margin -->|
     0  1  2  ...                                        ixg-1

    ixg = ix + 2*margin

配列 (``Bph`` など) はすべて ``(ixg, jxg)`` の大きさで、**そのうち
本物は真ん中だけ**\ 。だから診断量を取るときは必ず物理セルに限る。

.. code-block:: python

    m = grid.margin
    sl = (slice(m, grid.ixg - m), slice(m, grid.jxg - m))
    B = Bph[sl]                   # 物理セルだけ
    print(np.abs(B).max())

``slice(a, b)`` は ``a:b`` を変数として持ち回れるようにしたもの。
``Bph[sl]`` は ``Bph[m:ixg-m, m:jxg-m]`` と同じ。

.. warning::
   ゴーストセル込みで ``max()`` を取って、実際の 6 倍の磁場強度
   (1.5 T のところを 9 T) を報告したことがある。**診断は必ず ``sl``
   を通す。**

8.3 ``drr`` / ``drrm`` / ``drr2`` の区別
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

似た名前が 3 つあり、非一様格子では全部違う値になる。

.. list-table::
   :header-rows: 1
   :widths: 16 30 54

   * - 名前
     - 定義
     - 使うところ
   * - ``drr``
     - ``rrm[i+1] - rrm[i]``
     - セルの幅。**発散** (フラックスの差) を割るとき
   * - ``drrm``
     - ``rr[i] - rr[i-1]``
     - 隣り合うセル中心の距離。**面での勾配**\ を作るとき
   * - ``drr2``
     - ``rr[i+1] - rr[i-1]``
     - 2 セル幅。**セル中心での中心差分**\ を作るとき

一様格子では ``drr == drrm`` かつ ``drr2 == 2*drr`` になるので、
取り違えても「ちょうど 2 倍」という素直な形でしか出ない。逆に非一様
格子では 3 つとも別の値になる。論文再現の設定 (``grid_stretch = [2, 3]``\ )
で実際に見ると

.. code-block:: python

    >>> i = 60
    >>> grid.drr[i], grid.drrm[i], grid.drr2[i]     # [cm]
    (3.3457e8, 3.2807e8, 6.6823e8)
    >>> grid.drr2[i]/grid.drr[i]
    1.9973                                          # 2 ではない

``drr2`` を渡すべきところに ``drr`` を渡して :math:`B_\theta` が
2 倍になっていたことがある (10 節)。**エラーにはならない。**

8.4 ブロードキャスト
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

形の違う配列どうしの演算は、足りない軸を自動で伸ばして行われる。

.. code-block:: python

    >>> RR.shape, rr.shape
    ((112, 76), (112,))
    >>> (RR * rr).shape       # これはエラーになる (76 と 112 が合わない)
    >>> (RR * rr[:, None]).shape    # 縦ベクトルにしてから掛ける
    (112, 76)

``rr[:, None]`` は「``rr`` に長さ 1 の軸を足して ``(112, 1)`` にする」。
S2MFD では ``np.meshgrid`` で最初から 2 次元の ``RR`` / ``TH`` を
作ってあるので、この書き方はほとんど出てこない。


9. numba — Python のループを速くする
--------------------------------------------------

9.1 なぜ要るか
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Python の ``for`` ループは C や Fortran の 100 倍ほど遅い。108x72 の
格子を 60 年ぶん (数百万ステップ) 回すには到底足りない。

逃げ道は 2 つある。

1. **numpy でベクトル化する** — ``a[1:] - a[:-1]`` のようにループを
   書かずに済ませる。速いが、複雑な式では中間配列がたくさんできて
   メモリ帯域を食う。
2. **numba でコンパイルする** — ループをそのまま書いて、機械語に
   翻訳してもらう。**Fortran とほぼ同じ速さで、見た目は Python**\ 。

S2MFD の心臓部は 2 を採っている。だから ``physics_core.py`` は
「numpy らしくない」書き方に見える。

.. code-block:: python

    @njit
    def kernel(Bph, ..., dth):
        ixg, jxg = Bph.shape
        for i in range(1, ixg - 1):
            for j in range(1, jxg - 1):
                ...

**これは Fortran のコードを Python の文法で書いていると思ってよい。**
物理をやる人には、むしろこちらのほうが読みやすいはずである。

9.2 ``@njit`` の性質
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``@njit`` を付けた関数には制約が付く。

* **最初の呼び出しで数秒コンパイルされる**\ 。2 回目以降は速い。
  「1 ステップ目だけ遅い」のはこれ。
* **numpy 配列と数値しか扱えない。** 辞書、クラスのインスタンス、
  文字列などは基本的に渡せない。だから ``kernel`` の引数が 39 個に
  なる (``grid`` を渡せないので中身をばらして渡すしかない)。
* Python の関数を中から呼べない (``@njit`` の付いた関数は呼べる)。

9.3 ``prange`` と並列化
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from numba import prange

    @kernel()
    def flux(...):
        for i in prange(1, ixg - 1):     # ← range ではなく prange
            ...

``prange`` は「このループは並列に回してよい」という宣言。
``parallel=True`` でコンパイルされたときだけスレッドに展開され、
そうでなければただの ``range`` になる (3.2 節)。

.. important::
   **``prange`` にしてよいのは、繰り返しどうしに依存がなく、要素ごとに
   書き込むだけのループに限る。** これを守れば結果はスレッド数に
   よらずビット単位で同一になる (足す順序が変わらないから)。

   総和 (``cell_integral`` など) は **意図的に逐次のまま**\ にしてある。
   補正加算の順序が変わると質量・角運動量の保存が機械精度で壊れる。

9.4 ``fastmath``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``@njit(fastmath=True)`` は「浮動小数点の結合則を使ってよい」という
許可で、除算を逆数の掛け算に置き換えるなどの最適化が効く。速いが
**結果がビット単位では変わる**\ 。S2MFD では

* 計算カーネル: ``fastmath=True`` (約 17 倍速い)
* 参照実装 ``time_marching_reference``\ : ``fastmath=False``

の 2 つを持ち、テストで両者を突き合わせている。


10. 位置で当てず、名前で引く
--------------------------------------------------

この節だけは Python の説明ではなく、**このコードで実際に起きた事故**\ の
話である。同じ型の間違いを 2 回している。

**(1) 引数の位置**

.. code-block:: python

    # 間違い
    Brr, Bth = poloidal_mag(Aph, grid.RR, grid.sinTH, grid.drr, grid.dth)
    #                                                      ^^^ drr2 のはず

``drr`` と ``drr2`` は型も形も同じなのでエラーにならない。一様格子では
:math:`B_\theta` がちょうど 2 倍になり、ローレンツ力が過大になった。
しかも **同じ間違いを 2 つのファイルで独立にやっていた**\ 。

対策として、位置引数で呼ぶ入口をやめ、``grid`` を渡すだけの
:func:`~S2MFD.physics.physics_core.poloidal_from_potential` を作った。
テストが「本体と実行スクリプトが生のカーネルを直接呼んでいないこと」を
確認している (``tests/test_physics_core.py``\ )。

**(2) 配列の列の位置**

.. code-block:: python

    # 間違い
    off = h.shape[1] - len(qkeys)      # 列数から引き算で位置を当てる
    q_lambda = h[:, off + qkeys.index('Q_Lambda')]

履歴配列の列を「全体の列数から引き算して」求めていた。あとから列を
6 本足したときにずれ、``E_B`` より後ろの列を ``Q_Lambda`` として読んで
**値が 100 分の 1 になった**\ 。正しくは、保存してある列名の表を引く。

.. code-block:: python

    # 正しい
    idx = {str(k): i for i, k in enumerate(d['hist_cols'])}
    q_lambda = h[:, idx['Q_Lambda']]

``{k: v for ...}`` は **辞書内包表記**\ 。``enumerate(lst)`` は
``(0, lst[0]), (1, lst[1]), ...`` を順に返す。

.. important::
   **教訓は 1 つ。「位置を計算で当てる」のをやめ、名前で引く。**
   引数はキーワードで渡す。列は名前の表で引く。値が桁で外れていたら、
   計算式より先に「読んでいる場所が正しいか」を疑う。


11. 詰まったときにやること
--------------------------------------------------

**その 1: 対話で触る。** 読んで考えるより速い。

.. code-block:: python

    >>> import S2MFD
    >>> cfg = S2MFD.build_cfg('parameters/rempel06_paper.py')
    >>> grid = S2MFD.Grid.from_cfg(cfg)
    >>> vars(grid).keys()          # どんな引き出しがあるか
    >>> grid.drr[:5]               # 実際の値
    >>> help(S2MFD.Grid.from_cfg)  # docstring を読む

**その 2: 型と形を出す。** バグの半分はここで見つかる。

.. code-block:: python

    >>> type(cfg.margin), cfg.margin
    (<class 'int'>, 2)
    >>> Bph.shape, Bph.dtype
    ((112, 76), dtype('float64'))

**その 3: テストを読む。** ``tests/`` にはそのコードが
「何を満たすべきか」が日本語の docstring 付きで書いてある。仕様書として
読める。走らせるのは

.. code-block:: bash

    pytest tests/ -q
    pytest tests/test_physics_core.py -q -k poloidal    # 名前で絞る

**その 4: 小さく試す。** 解像度を落とせば数秒で回る。

.. code-block:: python

    cfg = S2MFD.build_cfg('parameters/rempel06_paper.py', ix=36, jx=24)

**その 5: 記録を読む。** ``doc/dev_records/`` に、実際に踏んだ間違いと
その切り分け方が日付順に書いてある。特に
``2026-08-23_mistakes.md`` は失敗の型を分類したもので、
同じ穴に落ちないために書かれている。


まとめ
--------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 出てくる書き方
     - ひとことで言うと
   * - ``class`` / ``self``
     - 引数の束。``x.f()`` は ``C.f(x)`` の略記
   * - ``@dataclass``
     - ``__init__`` の自動生成
   * - ``@classmethod``
     - 別の作り方 (``Grid.from_cfg``\ )
   * - ``@property``
     - 括弧のいらないメソッド
   * - ``@njit`` などの ``@``
     - ``f = deco(f)`` の略記
   * - デコレータを返す関数
     - 環境変数で全カーネルを一斉に切り替えるため
   * - クロージャ (関数を返す関数)
     - 分岐を内側ループから消して速くするため
   * - 辞書のキャッシュ
     - コンパイル結果を取っておくため
   * - ``lambda`` と表
     - 派生量の式を 1 か所に集めるため
   * - ``*`` と ``**``
     - 並びを位置引数に / 辞書をキーワード引数にばらす
   * - ``getattr(o, 'x', 既定)``
     - 古いパラメタファイルでも動くようにするため
   * - ``importlib``
     - 設定ファイルを Python として実行するため
   * - スライスは view
     - 境界条件を代入で書けるようにするため
   * - ``margin`` と ``sl``
     - 診断は物理セルだけに限るため
   * - ``prange`` / ``fastmath``
     - 並列化と高速化。保存量に関わる総和は逐次のまま

**言語としての難しさはここまでで打ち止め**\ である。この先で難しいのは
Python ではなく、差分法・境界条件・時間刻みの安定性のほうで、それは
:doc:`dynamic` と :doc:`verification` に書いてある。
