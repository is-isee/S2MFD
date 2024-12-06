Usage
=====

S2MFDでは、デフォルトのシミュレーションパラメタを用意しており、以下で実行できる。

.. code-block:: python

    import S2MFD
    S2MFD.run_simulation()

デフォルトのパラメタは `Jouve et al., 2008 <https://ui.adsabs.harvard.edu/abs/2008A%26A...483..949J/abstract>`_ のFlux transport dynamoの動径磁場上部境界条件のものである。

パラメタを変更するには、以下の二つの方法がある。

1. パラメタファイルを用意する。

   :code:`S2MFD.run_simulation()` の引数にパラメタファイルのパスを指定する。 :code:`S2MFD/parameters/` 以下にパラメタファイルを設置し、そのパスを指定する。

    .. code-block:: python


       import S2MFD
       S2MFD.run_simulation('parameters/your_parameter_file.py')

    パラメタファイルは以下のように記述する。

    .. code-block:: python

        from S2MFD.parameters.default import *
        m = 0

    
    :code:`S2MFD/parameters/default.py` にはデフォルトのパラメタが記述されている。これを継承して、必要なパラメタを変更する。

2. パラメタを直接指定する。

    パラメタサーベイをする場合などは、こちらで対応する。 :code:`S2MFD.Cfg` クラスを読み込んだ後に、直接パラメタを指定する。

    .. code-block:: python

        import S2MFD
        cfg = S2MFD.Cfg()
        cfg.m = 0
        S2MFD.run_simulation(cfg=cfg)
    