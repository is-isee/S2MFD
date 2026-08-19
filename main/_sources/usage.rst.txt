Usage
=====

基本の実行
----------

S2MFDでは、デフォルトのシミュレーションパラメタを用意しており、以下で実行できる。

.. code-block:: python

    import S2MFD
    sim = S2MFD.run_simulation()

デフォルトのパラメタは `Jouve et al., 2008 <https://ui.adsabs.harvard.edu/abs/2008A%26A...483..949J/abstract>`_ のFlux transport dynamoの動径磁場上部境界条件のものである。
:code:`run_simulation` は実行後の :code:`S2MFD.Simulation` オブジェクトを返す
(磁場 :code:`sim.Bph`, :code:`sim.Aph` や時刻 :code:`sim.time` にアクセスできる)。

出力は :code:`cfg.datadir` (既定 :code:`data/`) に、40日 (:code:`dtout`) ごとの
スナップショット :code:`data.NNNNNN.npz` (:code:`Bph, Aph, time, n, nd, dt, uu0, so0`)
と、メタデータ (:code:`config.json`, :code:`grid.npz`, :code:`setup.npz`,
:code:`legendre.npz`) が書かれる。

パラメタの変更
--------------

パラメタを変更するには、以下の二つの方法がある。

1. パラメタファイルを用意する。

   :code:`S2MFD.run_simulation()` の :code:`parameter_file` 引数にパラメタファイルのパスを指定する。パスが存在すればそのまま(絶対パス・カレント相対)、存在しなければ :code:`S2MFD/` パッケージディレクトリ相対として解決される。

    .. code-block:: python


       import S2MFD
       S2MFD.run_simulation(parameter_file='parameters/your_parameter_file.py')

    パラメタファイルは以下のように記述する。

    .. code-block:: python

        from S2MFD.parameters.defaults import *
        m = 0


    :code:`S2MFD/parameters/defaults.py` にはデフォルトのパラメタが記述されている。これを継承して、必要なパラメタを変更する。

2. パラメタを直接指定する。

    パラメタサーベイをする場合などは、こちらで対応する。 :code:`S2MFD.Cfg` クラスを読み込んだ後に、直接パラメタを指定する。

    .. code-block:: python

        import S2MFD
        cfg = S2MFD.Cfg()
        cfg.rey = 1400       # 基本量の変更
        cfg.resolve()        # 派生量 (uu0 = rey*ett/RSUN など) を再計算
        S2MFD.run_simulation(cfg=cfg)

    .. note::

        :code:`uu0` や :code:`so0` のような派生量は、基本量 (:code:`rey`,
        :code:`cso`, :code:`ett` など) の変更後に :code:`cfg.resolve()` を
        呼ぶことで再計算される (:code:`S2MFD.Simulation` の生成時にも自動で
        呼ばれる)。派生量を直接代入した場合は、その値が優先され resolve()
        では上書きされない。

物理モデルの選択肢
------------------

背景場は文字列スイッチで選択する。未知の文字列は :code:`ValueError` になる。
各選択肢が必要とするパラメタは :code:`S2MFD/setup.py` の該当分岐を参照のこと
(H10系のパラメタ例は :code:`parameters/hotta10.py` にある)。

.. list-table::
   :header-rows: 1

   * - スイッチ
     - 選択肢
     - 意味
   * - :code:`differential_type`
     - 'J08' / 'H10'
     - 差動回転プロファイル
   * - :code:`diffusive_type`
     - 'J08' / 'H10'
     - 磁気拡散プロファイル
   * - :code:`alpha_type`
     - 'BL' / 'normal' / 'H10'
     - α効果 (BL, H10 は非局所型)
   * - :code:`meridional_circulation_type`
     - 'J08' / 'D99' / 'H10'
     - 子午面流プロファイル
   * - :code:`boundary_condition_type`
     - 'vertical' / 'potential'
     - 上部境界条件

.. warning::

    振幅パラメタ (:code:`so0` 等) の規格化は選択肢間で統一されていない
    (例: 'BL' と 'normal' で異なる係数が掛かる)。選択肢をまたいで振幅を
    比較しないこと。

ランの継続 (リスタート)
-----------------------

:code:`cont_flag = True` (既定) のとき、既存の :code:`datadir` があると
最後のスナップショットから自動で再開する (「何も考えなければ続き」が仕様)。
再開時には既存の :code:`config.json` と現在の設定の整合性が検査され、
物理・格子パラメタが食い違う場合はエラーになる (:code:`tend`, :code:`dtout`
などの変更 = ランの延長は許容される)。新規にやり直す場合は別の
:code:`datadir` を指定するか、既存データを削除する。

時間依存パラメタ
----------------

子午面流振幅 :code:`uu0` と α効果振幅 :code:`so0` は、cfg に呼び出し可能な
属性を定義すると出力ステップごとに時間変化させられる。

.. code-block:: python

    cfg = S2MFD.Cfg()
    cfg.uu0_of_time = lambda t: 700.0 + 100.0 * np.sin(2*np.pi*t / (22*365*86400))
    S2MFD.run_simulation(cfg=cfg)

引数 :code:`t` はシミュレーション時刻 [秒]、返り値は振幅である。
パラメタ推定 (:doc:`inference`) はこの仕組みの上に構築されている。

解析スクリプト
--------------

:code:`ana/` に解析スクリプトがある。いずれもコマンドライン第1引数で
データディレクトリを指定できる (既定 :code:`../data/`)。

.. code-block:: bash

    python ana/ana.py data/                # 一括読み込み (対話解析の起点)
    python ana/butterfly_diagram.py data/  # 蝶形図
    python ana/sunspots_number.py data/    # 黒点数プロキシ時系列
    python ana/J08_test.py data/           # Jouve+2008 ベンチマーク比較
    python ana/check_profile.py parameters/hotta10.py  # 背景場プロファイル確認

数値の厳密さと速度
------------------

時間積分カーネルは既定で高速経路を使う。除算を逆数の乗算に置き換え、
numba の fastmath を有効にしているため、1 substep あたり倍精度 1 ULP 程度
(相対 ~2e-16) の丸めの違いが参照実装との間に生じる。磁気拡散を含む散逸系
なのでこの差は積分しても増幅せず、黒点数時系列の相関は 1.000000000000、
適応度は小数 10 桁まで一致することを確認している。

参照実装とビット一致させたい場合 (数値実験の再現性を厳密に確かめるとき等) は:

.. code-block:: python

    cfg = S2MFD.Cfg()
    cfg.exact_arithmetic = True   # 約3倍遅いが参照実装とビット一致
    S2MFD.run_simulation(cfg=cfg)

参照実装そのものは :code:`S2MFD.physics.time_marching_reference()` として
残してあり、テスト (:code:`tests/test_kernel_equivalence.py`) で
両者の等価性を常時検証している。

テスト
------

.. code-block:: bash

    pip install -e ".[dev]"
    pytest              # 通常テスト (数十秒)
    pytest -m slow      # 長時間ベンチマークテスト
