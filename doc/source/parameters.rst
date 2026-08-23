パラメタファイルと新しいモデルの足し方
==========================================

S2MFD の設定は **Python ファイル 1 枚**\ で与える。既存のファイルを
``import *`` で継承して差分だけ書く、というのが基本の形。

.. contents:: 内容
   :local:


設定の読み込み
--------------

.. code-block:: python

    import S2MFD

    cfg = S2MFD.build_cfg('parameters/rempel06_paper.py')   # 同梱のもの
    cfg = S2MFD.build_cfg('/path/to/mymodel.py')            # 自分のもの
    cfg = S2MFD.build_cfg('parameters/rempel06_paper.py',   # 値の上書き
                          ix=216, jx=144, alpha0=25.0)
    grid = S2MFD.Grid.from_cfg(cfg)

``'parameters/xxx.py'`` と書くと同梱の ``S2MFD/parameters/`` を読む。
絶対パスを渡せば任意のファイルを読める。

.. warning::
   **派生量はファイル読込時に確定する。**\  たとえば ``defaults.py`` は

   .. code-block:: python

       ett = 1.e11
       rey = 700
       uu0 = rey*ett/RSUN        # <- 派生量

   と書いてある。``build_cfg(..., ett=2e11)`` としても ``uu0`` は
   変わらない。派生量を変えたいときは派生量そのものを渡すこと。


既存のパラメタファイル
----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - ファイル
     - 内容
   * - ``defaults.py``
     - 既定値。**直接編集しない**\ 。ほかは全部これを継承する
   * - ``alpha_omega.py``
     - :math:`\alpha\Omega` ダイナモ (運動学的)
   * - ``hotta10.py``
     - Hotta & Yokoyama (2010) の設定
   * - ``rempel06.py``
     - Rempel (2006) の物理パラメタ。領域上端は 0.96
       :math:`R_\odot` (数値的に安定な既定)
   * - ``rempel06_paper.py``
     - :math:`\uparrow` を**論文どおり**\ の領域 (0.65-0.985
       :math:`R_\odot`、北半球、108 :math:`\times` 72、``margin=2``) に
   * - ``rempel06_2pt.py``
     - :math:`\uparrow` に 2 点集中の非一様格子を入れたもの
   * - ``inference.py``
     - GA によるパラメタ推定用


モデルを切り替えるスイッチ
--------------------------

背景場のプロファイルは ``*_type`` で選ぶ。分岐は :class:`S2MFD.Setup` に
ある。

.. list-table::
   :header-rows: 1
   :widths: 30 22 48

   * - スイッチ
     - 選べる値
     - 何を決めるか
   * - ``differential_type``
     - ``'J08'``, ``'H10'``
     - 差動回転のプロファイル (運動学的モードのみ)
   * - ``meridional_circulation_type``
     - ``'J08'``, ``'D99'``, ``'H10'``
     - 子午面流 (同上)
   * - ``diffusive_type``
     - ``'J08'``, ``'H10'``, ``'R06'``
     - 磁気拡散 :math:`\eta_t(r)`
   * - ``alpha_type``
     - ``'BL'``, ``'normal'``, ``'R06'``, ``'H10'``
     - :math:`\alpha` 効果の形
   * - ``dynamics``
     - ``'kinematic'``, ``'full'``, ``'hydro'``
     - 誘導方程式だけか、流体も解くか、磁場なしか
   * - ``boundary_condition_type``
     - ``'vertical'``, ``'potential'``, ``'R06'``
     - 磁場の動径境界条件
   * - ``rsst_type``
     - ``'const'``, ``'mach'``
     - 音速抑制の掛け方 (動力学モード)


新しいモデルを足す
------------------

例として「差動回転のプロファイルに新しい形 ``'H22'`` を足す」場合。

**1. 分岐を書く。** ``S2MFD/setup.py`` の該当メソッドに ``elif`` を足す。

.. code-block:: python

    def build_differential_rotation(self, cfg, grid):
        if cfg.differential_type == 'J08':
            ...
        elif cfg.differential_type == 'H22':
            # cfg から必要な値を読み、self.om に (ixg, jxg) を作る
            self.om = ...
        else:
            raise ValueError(
                f'unknown differential_type: {cfg.differential_type!r}')

**2. パラメタファイルを作る。** 既存を継承して差分だけ書く。

.. code-block:: python

    """Hotta et al. (2022) の差動回転プロファイル."""
    from S2MFD.parameters.hotta10 import *      # noqa: F401,F403

    differential_type = 'H22'
    my_new_parameter = 1.234

**3. テストを書く。** 最低でも次の 2 つ。

* プロファイルが期待する形になっているか (境界値・単調性・
  解析解との比較)
* **既存のモデルの結果が 1 ビットも変わっていないか**

.. code-block:: python

    def test_h22_does_not_change_j08():
        """新しい分岐を足しても既定の挙動が変わらないこと。"""
        cfg = make_cfg('parameters/hotta10.py')
        grid = make_grid(cfg)
        setup = S2MFD.Setup(cfg, grid)
        assert np.array_equal(setup.om, expected_om)

.. tip::
   **既存の挙動をビット一致で固定してから変える。** この
   リポジトリでは、キーワード引数への書き換えや ``ana/`` の添字修正など、
   「壊していないこと」をビット一致で確かめてから進めている
   (:doc:`verification`)。


よく使うパラメタ
----------------

動力学モード (Rempel 2006) で意味を持つもの。単位は **CGS**\ 。

.. list-table:: 領域と格子
   :header-rows: 1
   :widths: 30 70

   * - 名前
     - 意味
   * - ``ix``, ``jx``
     - 物理セル数 (動径、余緯度)
   * - ``margin``
     - ゴースト層の数。**SLD を使うなら 2 以上**\  (リミタが境界面の
       2 セル先を見る)
   * - ``rrmin``, ``rrmax``
     - 領域の内外半径 [cm]
   * - ``thmin``, ``thmax``
     - 余緯度の範囲。``thmax = pi/2`` で北半球のみ
   * - ``grid_stretch``
     - 0 で一様。>0 で ``grid_stretch_center`` 付近に点を集める。
       配列にすると複数箇所に同時に集中できる

.. list-table:: 物理
   :header-rows: 1
   :widths: 30 70

   * - ``om0``
     - 剛体回転の角速度 [rad/s]
   * - ``nu0``, ``kappa0``
     - 対流層の乱流粘性・熱伝導 [cm\ :sup:`2`/s]
   * - ``lambda0``, ``lambda_n``, ``lambda_tilt_deg``
     - :math:`\Lambda` 効果の振幅・緯度依存の次数・傾き
   * - ``eta_c``, ``eta_bc``, ``eta_cz``
     - 磁気拡散 (放射層・対流層底・対流層) [cm\ :sup:`2`/s]
   * - ``alpha0``
     - :math:`\alpha` 効果の振幅 [cm/s]
   * - ``alpha_quenching``
     - :math:`\alpha` クエンチングを掛けるか。**論文の非運動学的ランは
       False**
   * - ``magnetic_buoyancy``
     - 動径運動量に磁気圧勾配を含めるか (表 1 の列 4/6/8 は False)
   * - ``rsst_zeta``
     - 音速抑制の係数 (100 で音速 1/100)

.. list-table:: 数値
   :header-rows: 1
   :widths: 30 70

   * - ``sld_cs_factor``
     - 人工拡散 (SLD) の強さ。**論文に対応物がない数値パラメタ**\ 。
       小さいほど散逸が弱いが、粗い格子では発散する
   * - ``cfl_safety``
     - CFL 安全率。指定しなければ von Neumann 解析の中立点の 0.9 倍を
       自動で使う
   * - ``artificial_diffusion``
     - False で人工拡散を全部切る (対照実験用)
   * - ``magnetic_open_boundary``
     - 磁場の人工拡散フラックスを動径境界で開ける。反対称境界
       (``'R06'``) を使うときに必要
   * - ``cont_flag``
     - True (既定) なら既存の ``datadir`` があれば黙って再開する
