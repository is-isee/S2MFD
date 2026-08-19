# IDPA 統合設計書

- 日付: 2026-08-19
- 目的: IDPA(GA によるパラメタ推定)を S2MFD 本体のサブパッケージ `S2MFD.inference` として取り込む
- 方針(ユーザー決定): 段階的リファクタ移植。GA アルゴリズムと物理の挙動は論文(Shimizu & Hotta 2026)と同一に保ち、twin experiment で検証する。
- 関連文書: [本体レビュー](2026-08-19_code_review.md) / [IDPA解析](2026-08-19_idpa_analysis.md) / [作業ログ](worklog.md)

## 1. 決定事項の記録(2026-08-19 議論)

| 論点 | 決定 |
|---|---|
| 統合方式 | 段階的リファクタ移植(バグを直しながら移植、挙動は論文準拠を維持) |
| 本体バグ | 重要バグも一括修正 |
| 記録場所 | 本ディレクトリ `doc/dev_records/` |
| テスト | **test-first**: バグ修正前に現状挙動を固定する characterization テストを整備(ユーザー提案) |
| 出力時刻ずれ | 修正する(1ステップ分、既存データと差が出ることを明記) |
| CFL 式 | **現状維持**+前提コメント1行(η はスカラー、現行格子は Δr≪rΔθ で差1〜2%) |
| cont_flag | **デフォルト True 維持**(「何も考えなければ続ける」が意図した仕様)。再開明示+整合性チェックを追加 |
| ルート setup.py | 削除し pyproject.toml の [project] に集約。`S2MFD/setup.py`(物理)は改名しない |
| scipy_test.py | 削除(内容は引き継ぎ記録にメモ、git 履歴から復元可) |
| Cfg 派生パラメタ | `cfg.resolve()` 方式(基本量から派生量を再計算、Simulation 初期化時に自動呼出) |
| ana/ | 最小限自立化(import・datadir 補完、npz_edit の破壊的上書きは別名保存へ) |
| GA 変異 | 既定は論文どおり(非有界)。クリップ/σ下限はオプション引数 |
| reference/ | .gitignore に追加(論文 PDF はリポジトリに入れない) |
| doc/ | Sphinx ドキュメントも修正・充実(ユーザー追加要望) |

## 2. 新レイアウト

```
S2MFD/
  inference/
    __init__.py     # 公開API: run_two_stage, run_ga, GAConfig など
    genetic.py      # パラメタ非依存の GeneticAlgorithm/Chromosome(旧 GA_for_{u0,s0} を1本化)
    problem.py      # DynamoProblem: param_spec + 観測 + 窓 → 個体評価(旧 DefunctionProblem)
    metrics.py      # correlation/nmse/period_mse/mape 純関数(旧 judge1-5)
    observations.py # SILSO 読込・40日内挿・極小期検出(旧 make_time_series + detect_minimum)
    compare.py      # 結果比較プロット・数値出力(旧 OBS_compare)
    cli.py          # ドライバ(旧 IDPA.py)。argparse、--interactive で対話モード温存、--seed
    data/
      SN_Yearly.csv         # SILSO 原本(内挿はコードで実行)
      Apht_saved.npy        # スピンアップ初期場
      Bpht_saved.npy
tests/              # pytest(本体 + inference)
doc/dev_records/    # 本記録群
```

## 3. 本体側の拡張(Phase 3)

1. **save() 拡張**: npz に `nd`, `uu0`, `so0`, `dt` を追加(プレーン配列。IDPA の `allow_pickle=True` は持ち込まない)。
2. **時間依存パラメタの規約統一**: `cfg.uu0_of_time(t) -> float` / `cfg.so0_of_time(t) -> float`(存在すれば出力ステップ毎に評価)。IDPA の2重シグネチャ hasattr スニッフィングを廃止。GA はゲノムをクロージャに閉じて渡す。
3. **Setup 差分更新**: `Setup.update_flow(uu0)` / `update_alpha(so0)` — 毎回の全再構築(erf 群の再計算)を避ける。
4. **`Simulation.run_window(t_start, t_end, on_output=None)`**: 観測窓の積分+黒点数時系列記録(旧 defunction_main_loop の一般化)。
5. **`Simulation.spin_up(...)`**: 初期場ロード → 80年+極小期到達まで助走(旧 initial_for_OBS/LAST の統合。保存の有無はフラグ)。
6. **`sunspot_proxy(Bph, grid, cfg)`**(tools): γ·B_φ²(r=0.7R☉, θ=75°)、γ=5.8653520852 を1定義に。

## 4. Stage 間の受け渡し(コード生成の廃止)

- Stage1 終了時に `stage1_result.json`(推定係数 + メタデータ)を出力。
- Stage2 は JSON を読んで `cfg.uu0_of_time` をクロージャとして構成。`parameter_s0.py` の書き換えは行わない。
- `a0_s`(Stage1 で推定した α 定数)も JSON 経由で Stage2 の初期情報として保持(旧実装では捨てられていた)。

## 5. 並列評価の分離

- 個体 i・世代 g の評価は `output_dir/gen{g:03d}/ind{i:03d}/` で実施。
- 評価後、中間 npz は既定で削除(`--keep-intermediate` で保持)。ベスト個体の最終ランのみ全出力を保存。
- matplotlib は Agg 固定。CWD への PNG 書き出しは全廃(全て output_dir 配下)。
- ワーカー数は `min(g_num, os.cpu_count())` を明示。

## 6. 検証計画(Phase 5)

1. 単体テスト: metrics 純関数、GA 操作(SBX/変異/ASP、少数個体でのガード)、Cfg.resolve()。
2. J08 ベンチマーク: Phase 2 修正前後の比較(タイムスタンプ修正分の差は期待値として記録)。
3. **Twin experiment**: IDPA の `make_ground_truth.py` の真値で合成データ → Step1→Step2 推定 → 論文表4.1 水準(r≈0.97+、u0誤差~3%)の再現をシード固定で確認。
4. エンドツーエンド: CLI を短い世代数で実行し出力一式を確認。

## 7. 持ち込まないもの(理由つき)

- bisection 系 ~900行(`simulation.py:685-1286` ほか): GA 経路から呼ばれない旧世代手法。git 履歴(IDPA リポジトリ)に残る。
- `initial_for_defunction`: 旧 Fourier 規約(a0_s/b1_s/omega_s)の残骸。
- `judge`〜`judge5` のメソッド形態: metrics.py の純関数へ移設(obs/sim 逆転と MAPE ゼロ割を修正)。
- `allow_pickle=True`、`ana/`(IDPA側)、未使用 CSV、`_exec_prot_crossover`、ASP v2(実験定数 `0.9*0.881` 埋め込み)。
