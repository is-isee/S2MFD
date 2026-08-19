Installation
============

現在は、S2MFDは開発中のため、PyPIには登録されていない。
そのため、GitHubから直接インストールする必要がある。

開発のためには、editableモード + 開発用依存 (pytest) でインストールすることを推奨する。

.. code-block:: bash

    $ git clone https://github.com/is-isee/S2MFD.git
    $ cd S2MFD
    $ pip install -e ".[dev]"

これで、S2MFDがインストールされる。任意の場所からS2MFDをインポートすることができる。
依存関係 (numpy, scipy, numba, matplotlib) は :code:`pyproject.toml` に
宣言されており、自動でインストールされる。

利用のみであれば通常のインストールでもよい。

.. code-block:: bash

    $ pip install .