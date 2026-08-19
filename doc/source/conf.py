# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../../'))
import S2MFD

project = 'S2MFD'
copyright = '2024, S2MFD project'
author = 'Hideyuki Hotta'

version = S2MFD.__version__
release = S2MFD.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# extensions = []
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # Google / NumPy スタイルの docstring サポート
    'sphinx.ext.viewcode',  # ソースコードのリンク
    'sphinx.ext.intersphinx',  # 他のドキュメントへのリンク
    'sphinx.ext.autosummary',
    'sphinx_automodapi.automodapi',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'numba': ('https://numba.readthedocs.io/en/stable/', None),  # Numba のドキュメントへのリンクを追加
}

templates_path = ['_templates']
exclude_patterns = []

language = 'en'

autosummary_generate = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'sphinx_book_theme'
#html_theme = 'alabaster'
#html_theme = 'sphinx_material'
html_theme = 'sphinx_rtd_theme'
#html_theme = "pydata_sphinx_theme"
html_static_path = []

autodoc_member_order = 'groupwise'
autodoc_default_options = {
    'special-members': '__init__',
    'private-members': False,
}

html_context = {
    "current_version": "main",  # 現在のバージョン
    "versions": [               # 他のバージョンリスト
        ("main", "/main/"),
    ],
}
# NOTE: 以前は sphinx-multiversion を使っていたが、Sphinx 8 と非互換のため廃止。
# バージョン別の公開 (gh-pages の main/, vX.Y.Z/) は GitHub Actions
# (.github/workflows/docs.yml) がデプロイ先ディレクトリを切り替えて実現している。

napoleon_include_init_with_doc = True
napoleon_use_ivar = True

automodapi_inheritance_diagram = False