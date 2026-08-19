Parameter Inference (GA)
========================

:code:`S2MFD.inference` は、遺伝的アルゴリズム (GA) による磁束輸送ダイナモ
モデルのパラメタ推定パッケージである。手法の詳細は
`Shimizu & Hotta (2026), ApJ 996, 102 <https://doi.org/10.3847/1538-4357/ae2855>`_
を参照。**フルスケール実行の実務手順 (ジョブの流し方・ログと出力の読み方・
トラブルシューティング) は** :doc:`tutorial_inference` **を参照。**

手法の概要
----------

- 推定対象: 子午面流振幅 :math:`u_0(t)` と Babcock-Leighton α効果振幅
  :math:`s_0(t)` の時間変化。それぞれ「線形関数 + sin級数 (l = 1, 2, 4)」の
  5係数で表現する (論文式 2.10)。
- 適応度: :math:`F = 0.9\,r - 0.1\,e` (黒点数時系列の相関係数 − 正規化二乗誤差、
  論文式 3.1)。40日間隔でサンプリングした黒点数プロキシ
  :math:`\mathrm{SSN} = \gamma B_\phi^2(0.7R_\odot, 75^\circ)` を観測
  (SILSO 年平均黒点数、パッケージに同梱) と比較する。
- 2段階推定: 同時推定は局所解に陥るため、Step 1 で :math:`u_0(t)`
  (:math:`s_0` は定数扱い)、Step 2 で :math:`s_0(t)` (:math:`u_0` は既知) の
  順に推定する。
- GA: 個体数30・最大50世代、SBX交叉 (:math:`\eta=2`, 80%)、ガウス変異
  (:math:`\sigma = 0.1|x|`, 30%)、ASPトーナメント選択 (適応度停滞と多様性に
  応じてトーナメントサイズ 3/7/5 を切替)、エリート保存。
- 各個体の評価: Jouve+2008 平衡場から開始振幅で80年+黒点数極小期まで
  スピンアップした後、観測窓を積分して比較する。

コマンドラインからの実行
------------------------

.. code-block:: bash

    # 観測データの極小年を確認 (推定窓は極小年から始める)
    python -m S2MFD.inference.cli --list-minima

    # 1944-1996 年を推定 (論文設定: 個体数30, 最大50世代)
    python -m S2MFD.inference.cli --start-year 1944 --end-year 1996 \
        --pop 30 --generations 50 --seed 42

    # 対話モード (旧 IDPA.py 互換)
    python -m S2MFD.inference.cli --interactive

主なオプション:

- :code:`--data`: 'OBS' (同梱の SILSO 年平均) または自前の CSV
  (1列目=年, 2列目=黒点数, ヘッダ1行)。40日間隔への内挿は自動で行われる。
- :code:`--stage both|u0|s0`: 実行する段階。Step 2 のみ実行する場合は
  Step 1 の結果 (:code:`stage_result.json`) が必要。
- :code:`--fitness standard|period`: 適応度。:code:`period` は周期の
  平均二乗誤差ベース (ダルトンミニマムなどの急変期用、論文式 4.2)。
- :code:`--seed`: 乱数シード (再現可能な実行)。
- :code:`--workers`: 並列評価のプロセス数 (既定: min(個体数, CPU数))。

出力は :code:`results/datas_{開始}_{終了}/data_{u0,s0}_{開始}_{終了}/` に:

- :code:`stage_result.json` — 推定パラメタ・適応度・GA履歴・設定の一式
- :code:`parameters.txt` — 推定パラメタ (旧 IDPA 互換形式)
- :code:`P_sunspots_number_compare.png` / :code:`P_u0_compare.png` /
  :code:`P_s0_compare.png` / :code:`split_functions_now.png` — 比較プロット
- :code:`fitness_time.png` / :code:`diversity_time.png` — GA の収束履歴
- :code:`data.NNNNNN.npz` — 最良個体の再シミュレーションのスナップショット
- :code:`numerical_result.txt` — 相関係数・NMSE・適応度

既に :code:`stage_result.json` がある段階はスキップされる (レジューム)。

Python API からの実行
---------------------

.. code-block:: python

    from S2MFD.inference import (
        load_observations, DynamoProblem, GeneticAlgorithm, GAConfig)
    from S2MFD.inference.observations import load_initial_field

    obs = load_observations('OBS')
    Bph0, Aph0 = load_initial_field()
    i0, i1 = obs.index_of_year(1944), obs.index_of_year(1996)

    problem = DynamoProblem(
        'parameters/inference.py', obs.seconds, obs.ssn,
        i0, i1, mode='u0', initial_Bph=Bph0, initial_Aph=Aph0)

    ga = GeneticAlgorithm(problem, problem.param_spec(),
                          population_size=30,
                          config=GAConfig(seed=42))
    result = ga.run()
    print(result.best_params, result.best_fitness)

Twin experiment (合成データによる検証) は、真値パラメタで
:code:`problem.simulate(true_params)` を実行して得た時系列を観測の代わりに
使えばよい。真値の適応度は定義上 0.9 (相関1, NMSE 0) になるため、
パイプラインの健全性チェックにも使える。

計算コストの目安
----------------

1個体の評価は約3万ステップ (80年スピンアップ + 50年窓、128×128格子) で、
1コアあたり数分程度。論文設定 (個体数30×50世代) はマルチコアで数時間〜の
計算になる。開発・検証には :code:`spinup_years` や窓の長さを縮めた設定が
有効 (:code:`DynamoProblem(spinup_years=..., until_minimum=...)`)。
