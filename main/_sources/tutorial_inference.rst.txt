Tutorial: パラメタ推定を実際に回す
==================================

このページは、GA パラメタ推定 (:doc:`inference`) を**論文設定のフルスケールで
実行する実務手順**をまとめたチュートリアルである。手法・API の説明は
:doc:`inference` を、手法の背景は Shimizu & Hotta (2026) を参照。

0. 準備
-------

.. code-block:: bash

    cd S2MFD
    pip install -e ".[dev]"
    pytest -q          # まず全テストが通ることを確認 (~10秒)

計算資源の目安:

- 1個体の評価は約3万ステップ (80年スピンアップ + 50年窓、128×128格子) で、
  **1コアあたり2〜3分** (ウォーム時 ~5 ms/ステップ)
- 論文設定 (個体数30 × 最大50世代 = 最大1500評価) の所要時間は
  **8コア並列で半日程度**。世代数・窓を減らせば線形に短くなる。
- メモリは1ワーカーあたり数百MB程度。個体数 ≦ コア数が効率的
  (:code:`--workers` で明示可)。

1. 推定窓を決める
-----------------

推定は黒点数の**極小年から極小年まで**の区間で行う (初期条件が極小期の状態
で作られるため)。まず極小年の候補を確認する:

.. code-block:: bash

    $ python -m S2MFD.inference.cli --list-minima
    検出された極小年: [1711.6 1723.5 1733.5 ... 1944.4 1954.4 1964.5 1976.4 1986.4 1996.5 ...]

論文では約5周期 (50〜60年) ごとに区間分割している (例: 1723–1775, 1775–1833,
1833–1889, 1889–1944, 1944–1996, 1996–2024)。60年を超える指定はエラーになる。

2. フルスケール実行
-------------------

長時間ジョブなので nohup や screen/tmux で流す:

.. code-block:: bash

    nohup python -m S2MFD.inference.cli \
        --start-year 1944 --end-year 1996 \
        --pop 30 --generations 50 --target-fitness 0.80 \
        --seed 42 --output results \
        > inference_1944_1996.log 2>&1 &

    tail -f inference_1944_1996.log   # 進捗を眺める

オプションの考え方:

- :code:`--seed` は**必ず指定する**(再現可能な実行のため。値は何でもよい)。
- :code:`--target-fitness 0.80` は論文の推奨値。到達したらその世代で打ち切り、
  到達しなければ :code:`--generations` まで回って最良個体を採用する。
- ダルトンミニマム (1798–1833 など急変区間) では
  :code:`--fitness period` を使う (論文 4.2 章と同じ切替)。

3. ログの読み方
---------------

.. code-block:: text

    ... INFO: === Stage u0: GA開始 (個体数30, 最大50世代) ===
    ... INFO: generation 12: best fitness=0.834502 diversity=0.1873
    ... INFO: best params: {'as_u': 812.3, 'ae_u': 745.1, ...}

- :code:`generation N: best fitness=...` — その世代までの歴代最良の適応度。
  **エリート保存により単調非減少**。しばらく上がらない世代が続くのは正常
  (ASP 選択が探索モードに切り替わっている)。
- :code:`diversity` — 個体群の多様性 (0に近いほど収束)。
  適応度が停滞したまま diversity も小さいときは局所解の可能性がある
  → seed を変えてもう1本流して比較するのが実務的。
- 1世代あたりの所要時間は最初の世代で分かる (JITコンパイルの分だけ
  初世代は遅い)。全体時間 ≈ 世代時間 × 世代数で見積もれる。

4. 中断と再開
-------------

- **Ctrl-C (中断)**: その時点の最良個体で最終処理 (再シミュレーション・保存)
  まで行って終了する。:code:`stage_result.json` に
  :code:`"interrupted": true` が記録される。
- **再実行**: 同じ :code:`--output` で再度実行すると、
  :code:`stage_result.json` が存在する Stage はスキップされる。
  Stage 1 完了後に落ちた場合は、そのまま再実行すれば Stage 2 から続く。
- **Stage 1 をやり直したい**: 該当ディレクトリ
  (:code:`results/datas_1944_1996/data_u0_1944_1996/`) を削除して再実行。
- Stage 2 だけ流す: :code:`--stage s0` (Stage 1 の
  :code:`stage_result.json` が必要)。

5. 出力の読み方
---------------

:code:`results/datas_{開始}_{終了}/data_{u0,s0}_{開始}_{終了}/` に以下が出る:

.. list-table::
   :header-rows: 1

   * - ファイル
     - 中身と見方
   * - :code:`stage_result.json`
     - **結果の本体**。推定パラメタ (params)、GA終了時の適応度 (fitness)、
       最終ランの再評価 (rerun_scores: cc=相関, nmse, eva)、世代ごとの
       適応度/多様性履歴、実行設定 (config: seed含む)。
   * - :code:`parameters.txt`
     - 推定パラメタの1行テキスト (旧 IDPA 互換)。u0 は
       as_u, ae_u, a1_u, a2_u, a4_u, a0_s の順。
   * - :code:`P_sunspots_number_compare.png`
     - 黒点数の観測 (赤) vs 推定 (青)。**まず見るべき図**。周期の位相が
       合っているか、振幅の大小関係が再現されているかを確認する。
   * - :code:`P_u0_compare.png` / :code:`P_s0_compare.png`
     - 推定された u0(t), s0(t)。論文の物理的解釈では u0 は −51%〜+29%、
       s0 は −69%〜+83% 程度の変動幅だった。極端に範囲外なら疑う。
   * - :code:`split_functions_now.png`
     - 推定関数の成分分解 (線形 + l=1,2,4 の sin)。どの成分が効いているか。
   * - :code:`fitness_time.png` / :code:`diversity_time.png`
     - GA の収束履歴。fitness が頭打ちになってから十分な世代が
       経っているか (=収束したか) の確認。
   * - :code:`data.NNNNNN.npz`
     - 最良個体の再シミュレーションのスナップショット
       (番号は観測インデックス)。:code:`ana/` スクリプトで蝶形図等を描ける。
   * - :code:`numerical_result.txt`
     - cc / NMSE / EVA の追記ログ (実行日時つき)。

**良し悪しの目安** (論文の値):

- 通常区間: 相関 cc ≈ 0.87〜0.97、NMSE ≈ 0.02〜0.10
- ダルトンミニマム区間は難しい (論文でも cc = 0.685)
- twin experiment (合成データ) なら理論最大 fitness = 0.9。
  0.89 台に届かない場合は世代数・個体数を増やす。

6. 自前の観測データを使う
-------------------------

1列目=年 (小数可)、2列目=黒点数、1行目ヘッダの CSV を用意する
(40日間隔への内挿は自動):

.. code-block:: bash

    python -m S2MFD.inference.cli --data my_ssn.csv --list-minima
    python -m S2MFD.inference.cli --data my_ssn.csv --start-year ... --end-year ...

列が違う場合は Python API で :code:`load_observations(path, time_col=..., ssn_col=...)`
を使う。

7. 検証 (twin experiment) のやり方
----------------------------------

コード変更後の妥当性確認には、真値パラメタから合成した観測での回収試験を行う。
実行例スクリプトが :code:`doc/dev_records/verification/twin_experiment.py`
(縮小版、seed固定) にある:

.. code-block:: bash

    python doc/dev_records/verification/twin_experiment.py   # ~15分
    python doc/dev_records/verification/twin_eval.py         # 論文式4.1の時系列誤差を表示

チェックポイント:

1. **真値パラメタの適応度がちょうど 0.9** になること (パイプラインの健全性)。
   ここがずれたらコードのどこかが壊れている。
2. GA の適応度が世代とともに 0.9 に向かうこと。
3. 時系列誤差 (式4.1) が予算に応じて縮むこと (論文フル設定で u0 ≈ 3%)。

8. トラブルシューティング
-------------------------

- **ログに「individual failed ... fitness = -inf」**: その個体のパラメタで
  シミュレーションが数値的に破綻した (NaN 検知)。少数なら GA が自然に
  淘汰するので問題ない。頻発する場合は初期範囲・変異が極端な値を出して
  いないか確認 (:code:`--clip-mutation` で範囲内に制限できる)。
- **「spin_up: no sunspot-number minimum detected」**: スピンアップ解が
  発振していない (ダイナモが減衰する等)。パラメタ範囲が臨界以下の設定に
  なっていないか確認。
- **1世代が想定より遅い**: ワーカー数がコア数を超えていないか、
  他のジョブと競合していないか。:code:`--workers` で調整。
- **結果が毎回変わる**: :code:`--seed` を指定していない。指定していれば
  ワーカー数によらず同一ゲノム列になる (評価は決定論的)。
- **旧 IDPA の結果と数値が完全一致しない**: 乱数系列の変更(numpy Generator化)と
  出力タイミング修正のため、個々のランは一致しない。統計的な水準
  (適応度・誤差) で比較すること。詳細は
  :code:`doc/dev_records/handover.md` の「意図的に挙動を変えた点」。
