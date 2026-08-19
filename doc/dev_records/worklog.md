# 作業ログ(IDPA 統合プロジェクト)

新しいエントリを上に追記する。各エントリは「何を・なぜ・どう検証したか」を記録する。

---

## 2026-08-19

### Phase 4: `S2MFD/inference/` サブパッケージ新設(完了)

IDPA の GA 層を新サブパッケージとして移植(92 passed):

- `observations.py` — SILSO 年平均(原本 `data/SN_Yearly.csv` を同梱)の読み込み・40日内挿(旧 make_time_series の関数化)・極小期検出(旧 detect_minimum)。派生物だった `SN_Yearly_interp.csv` はコミットせずコードで生成。初期磁場 `data/{A,B}pht_saved.npy` も同梱し importlib.resources で解決。
- `metrics.py` — 旧 judge1〜5 を純関数化(correlation/nmse/total_count_error/period_mse/mape/fitness)。**judge3 の obs/sim 逆転バグを修正**(コード内 TODO で自認されていたもの)。MAPE のゼロ割を eps ガード。
- `genetic.py` — GA_for_u0/GA_for_s0(98%重複)を param_spec 引数の1実装に統合。**シード指定で再現可能**(numpy Generator 統一)。**個体数<10 のクラッシュを解消**(トーナメントサイズは論文どおり 3/7/5 固定 + ガード。個体数30では旧実装の len//10 等と同値)。エリートは固定スロット3ではなくランダム置換。クリップ変異・σ下限はオプション(既定は論文どおり非有界)。KeyboardInterrupt は初世代中でも安全に途中結果を返す。print → logging。
- `problem.py` — DynamoProblem(評価器)。**個体評価は完全ディスクレス**(save_dir=None)にして並列出力競合を根絶。発散個体は例外でなく fitness=-inf。時間変化関数 linear_plus_sines(論文式2.10)と picklable な TimeFunction。fitness_kind='period'(ダルトンミニマム用、式4.2)も選択可。
- `compare.py` — 旧 OBS_compare の関数化(パス注入、Agg固定)。GA履歴プロットも。
- `cli.py` — 旧 IDPA.py の argparse 版。`--interactive` で対話モード互換、`--seed`、`--list-minima`、`--stage both|u0|s0`。**Stage1→Stage2 の受け渡しは stage_result.json**(ソースコード書き換えを廃止。旧実装で捨てられていた a0_s も保存)。既存結果があればスキップ(旧互換のレジューム)。旧互換の parameters.txt も出力。
- `parameters/inference.py` — 推定用設定(defaults + potential 境界)。

検証: 全92テスト(metrics・TimeFunction 端点性質・GA収束/再現性/少数個体/並列・観測データ・128×128 実格子での1個体評価スモーク)。CLI --list-minima の極小年は論文の区間境界(1944, 1996 等)と一致。

### Phase 3: Simulation への GA 拡張API追加(完了)

IDPA の生きているコード(~250行)を汎用APIとして再設計して本体に追加(71 passed):

1. **save() 拡張** — npz に `nd`, `dt`, `uu0`, `so0` を追加(プレーン float/int なので IDPA の `allow_pickle=True` は不要)。`cfg.verbose = False` で進捗 print を抑制可能に(並列GA用)。
2. **時間依存パラメタの規約統一** — `cfg.uu0_of_time(t)` / `cfg.so0_of_time(t)`(t は絶対シミュレーション時刻 [s])という単一のコールバック規約を新設。`update_time_dependent_parameters()` が出力ステップ毎(main_loop / run_window / initial_condition)に評価する。IDPA の2重シグネチャ hasattr スニッフィング(`so0_time_dependent` の (time,ett,RSUN) 版と (as_s,...,inf_year,time) 版の衝突)を解消。
3. **Setup のビルダー分割** — `build_rotation/build_diffusivity/build_alpha/build_flow` に分割し、`__init__` は4つを順に呼ぶだけに。時間依存更新では該当プロファイルのみ再構築(IDPA は毎回 Setup 全再構築+erf 群再計算をしていた)。数値は同一(式は移動しただけ)。
4. **`run_window(t_end, on_output=None, save_output=True)`** — 観測窓ループ(旧 defunction_main_loop の一般化)。コールバックで黒点数時系列を記録、`save_output=False` でディスク出力なしの個体評価が可能。`run_window(tend)` は main_loop とビット同一の結果(テストで確認)。
5. **`spin_up(duration, until_minimum=True, proxy=None)`** — 助走計算(旧 initial_for_OBS/LAST の統合)。3点履歴による黒点数極小検出、`max_extra` の打ち切り付き(旧実装は発振しない解で無限ループ)。`set_field()` で初期場を投入する設計にし、`initial_data/*.npy` のハードコードパスを排除。
6. **`tools.sunspot_proxy(Bph, grid, cfg, gamma=5.8653520852)`** — 黒点数プロキシを1定義に(IDPA では7箇所にコピペ)。

**IDPAとの意図的な差異**: 時間依存パラメタの更新タイミングが1ステップ分ずれる(Phase 2 のループ順序修正の帰結。40日出力間隔に対し dt≈1.6日)。twin experiment(Phase 5)で論文水準の再現を確認する。judge1〜5 は Phase 4 で metrics.py の純関数として移植する(Simulation には持ち込まない)。bisection 系 ~900行は持ち込まない。

### Phase 2: S2MFD 本体の重要バグ修正(完了)

実施した修正(すべて test-first、60 passed / slow 2 passed):

1. **出力タイムスタンプずれ修正** — `main_loop` を「積分 → 時刻更新 → 出力」の順に変更。スナップショットのラベルと場が一致するようになった。**ラン終端の最終状態は旧実装とビット同等**(検証: 旧ゴールデンとの最大相対差 1.4e-14、これは grid の linspace 化による丸め差のみ)。差が出るのは各スナップショットの中身(旧: 1ステップ前の場)だけ。
2. **CFL条件** — 式は現状維持(ユーザーと議論済み)。前提コメント(Δr ≪ rΔθ)を追加し、純Python二重ループを numpy にベクトル化(値は同一、`test_matches_vectorized_formula` で確認)。
3. **0次元配列問題** — `S2MFD/npz_io.py` の `NpzIO` mixin を新設し Grid/Setup/Legendre の save/load 3重複を集約。load 時に `.item()` でスカラー型を復元。`data_load` も float()/int() で復元。`allow_pickle=True` は廃止(プレーン配列のみなので不要)。
4. **cont_flag 再開安全化** — デフォルト True は維持(意図した仕様)。再開時に「Resuming existing run…」を明示表示し、既存 config.json との整合性チェックを追加(tend/dtout 等の「再開時に変えてよいキー」以外の不一致は RuntimeError)。再開時の Legendre は npz からではなく grid から再構築(grid の純関数のため。旧フォーマット npz との互換問題も回避)。
5. **パッケージング修復** — `pyproject.toml` の `[project]` に集約(dependencies 宣言、packages.find でサブパッケージ包含)、ルート `setup.py` 削除、`pytest.ini` を `[tool.pytest.ini_options]` に統合。wheel にサブパッケージが全て入ることを確認。requirements.txt は `-e .[dev]` の1行に。
6. **Cfg 改善** — `resolve()` 新設(派生量の再計算。パラメタファイルが導出式と異なる値を明示した名前と、ユーザーが直接代入した名前は pin して上書きしない)。Simulation 生成時に自動呼出。パラメタファイルの絶対パス/パッケージ外パス受け入れ。save() で numpy スカラーをネイティブ型に変換。パス結合を os.path.join に統一。
7. **NaN 早期検知** — `check_finite()` を出力ステップ毎に呼び、非有限値で RuntimeError。
8. **run_simulation が Simulation オブジェクトを返す**ように変更。
9. **デッドコード削除** — `scipy_test.py`(426行)、`simulation.py` の誤った `__all__`・save() 内の未使用 poloidal_mag 計算・main_loop の未使用 matplotlib import とコメントアウトされた描画コード、各所の未使用 import。`__init__.py` の未使用 `paramdir`。setup.py の @dataclass 誤用を除去し docstring に `so` を追記。*_type 文字列スイッチ全てに else: raise ValueError を追加。tools.py の docstring を実装に一致させ(up/dw)、エラー分岐も raise に変更。
10. **性能改善** — alpha_effect の np.repeat → ブロードキャスト、potential BC の時間不変量(sinth, n_values, coefficients, P1n_reduced)を Legendre.__init__ に事前計算、grid 座標を linspace 相当に(逐次加算の丸め蓄積を解消)。
11. **ana/ 最小限自立化** — 共通ローダ `ana/ana_common.py` を新設(datadir をコマンドライン引数で指定可)。全スクリプトが単体実行可能に。`npz_edit.py` は破壊的上書きをやめ `*_edited.npz` に保存。`sunspots_number.py` の時間軸を実スナップショット時刻に修正。`check_profile.py` のハードコードインデックス(so[:,43] 等)を角度・半径からの計算に変更。動作確認済み(小規模ラン + ヘッドレス実行)。
12. **doc/source/usage.rst の誤記修正** — 位置引数の誤用例(クラッシュする)、default.py → defaults.py、resolve() の説明を追加。

ゴールデンデータは Phase 2 完了時点の挙動で再生成した(旧版との差は上記1のとおり)。

### Phase 1: テスト基盤の先行整備(完了)

- 開発環境: リポジトリ直下に `.venv` を作成(numpy 2.5.2 / numba 0.67.0 / scipy 1.18.0 / pytest 9.1.1)。requirements.txt のピン(numpy==1.24.3 等)は Python 3.12 と非互換のため現行版を使用 — Phase 2 の pyproject 集約で依存宣言を更新する。
- `tests/` を新設(pytest、`pytest.ini` は Phase 2 で pyproject に統合予定):
  - `test_tools.py` — 微分演算子。**docstring の up/dw が実装と逆であることをテストで文書化**(実装挙動を正とする)。保存形ペア('up'→'dw' 合成 = 2階中心差分)も検証。
  - `test_grid.py` — セル中心座標・ゴースト・面中心座標・save/load。スカラーの0次元 ndarray 化は xfail で記録。
  - `test_cfg.py` — defaults/派生量/上書き/JSONラウンドトリップ。既知の制限3件(派生量の固定化、numpy型の黙殺、絶対パス不可)を xfail/characterization で記録。
  - `test_setup_profiles.py` — J08 のプロファイル性質 + defaults/alpha_omega/hotta10 のゴールデン配列比較。
  - `test_physics_core.py` — 双極子場の poloidal_mag、一様場・一様流の移流(厳密解)、αクエンチング、BL非局所性。
  - `test_boundary_condition.py` — vertical/potential のゴースト関係・線形性・n=1モードの外挿一致。
  - `test_simulation.py` — CFL式のcharacterization、32×32小規模ラン(alpha_omega, 200日)のゴールデン比較、**タイムスタンプずれの xfail テスト**(修正後にパスする形で記述)、リスタート回帰テスト、`run_simulation` API。
  - `test_slow_benchmark.py` — `-m slow` でのみ実行(αΩダイナモ成長、potential BC 安定性)。
  - ゴールデンデータは `tests/golden/generate_golden.py` で生成(再生成時は本ログに記録すること)。
- 結果: **46 passed / 6 xfailed / slow 2 passed**。
- **発見**: リスタート経路は「0次元配列問題で壊れている可能性が高い」と調査段階で推定していたが、**現行の numpy/numba では正常動作する**ことをテストで確認(`test_restart_continues_run` は通常のパステストに変更)。型の汚れ自体は残っており Phase 2 で健全化する(レビュー報告書 §1.2 の深刻度は「実行可否」ではなく「型健全性・将来の互換性」に読み替え)。

### 調査・計画(Phase 0 まで)

- **調査**: S2MFD 全ソース精読、IDPA 全ソース精読+upstream との diff、修論 PDF(reference/)全55ページの要約を実施。結果は以下に整理:
  - [2026-08-19_code_review.md](2026-08-19_code_review.md) — 本体レビュー(重大7件・設計7件・品質・性能)
  - [2026-08-19_idpa_analysis.md](2026-08-19_idpa_analysis.md) — IDPA 解析(バグ16件・差分表・検証基準)
  - [2026-08-19_integration_design.md](2026-08-19_integration_design.md) — 統合設計(決定事項13件を含む)
- **ユーザーとの議論で決定**(詳細は統合設計書 §1):
  - CFL 式は現状維持(η スカラー・Δr≪rΔθ で実害なしと確認)。コメントのみ追加。
  - cont_flag=True 維持(意図した仕様)。整合性チェックのみ追加。
  - test-first で進める(ユーザー提案)。
  - タイムスタンプずれは修正、ルート setup.py 削除、scipy_test.py 削除、Cfg.resolve() 方式、ana/ 最小限自立化、変異は論文どおりを既定に。
- **ブランチ**: `feature/idpa-integration` を `main`(6cc3846)から作成。
- **リポジトリ整理**:
  - `.gitignore` に `reference/` を追加(論文 PDF 置き場。ユーザー要望)。
  - ディスク上に残っていた `__pycache__/` を削除(git 未追跡、.gitignore 済みだった)。
- **記録基盤**: `doc/dev_records/` を新設し本記録群を作成(Phase 0 完了)。

### 未着手(次回以降)

- Phase 1: pytest テスト基盤(characterization テスト)
- Phase 2〜6: 統合設計書のとおり
