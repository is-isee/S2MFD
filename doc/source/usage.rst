Usage
=====

S2MFDでは、デフォルトのシミュレーションパラメタを用意しており、以下で実行できる。

.. code-block:: python

    import S2MFD
    S2MFD.run_simulation()

デフォルトのパラメタは `Jouve et al., 2008 <https://ui.adsabs.harvard.edu/abs/2008A%26A...483..949J/abstract>`_ のFlux transport dynamoの動径磁場上部境界条件のものである。

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
