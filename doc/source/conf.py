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
release = '2024.11.28'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# extensions = []
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # Google / NumPy スタイルの docstring サポート
    'sphinx.ext.viewcode',  # ソースコードのリンク
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),  # NumPy のドキュメントへのリンクを追加
    # 他のプロジェクトのドキュメントへのリンクを追加する場合はここに記述
}

autodoc_member_order = 'groupwise'
autosummary_generate = True  # 自動要約テーブルの生成を有効にする

templates_path = ['_templates']
exclude_patterns = []

language = 'en'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
#html_theme = 'alabaster'
#html_theme = 'sphinx_material'
#html_theme = 'sphinx_rtd_theme'
html_static_path = []

# napoleon_google_docstring = False  # Googleスタイルを無効化（NumPyスタイルのみ使用）
# napoleon_numpy_docstring = True   # NumPyスタイルを有効化
# napoleon_include_init_with_doc = True  # __init__メソッドのdocstringをクラスdocstringに含めるか
# napoleon_include_private_with_doc = False  # プライベートメソッドのdocstringを含めるか
# nanolean_use_attribute = True  # 属性のドキュメントを生成するか
# napoleon_use_param = True  # パラメータリストをSphinxの:paramに変換
# napoleon_use_rtype = True  # 戻り値をSphinxの:returnに変換