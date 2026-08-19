# 引き継ぎ書(IDPA 統合プロジェクト)

- 日付: 2026-08-19
- 作業ブランチ: `feature/idpa-integration`(`main` の 6cc3846 から分岐)
- 詳細な作業経緯: [worklog.md](worklog.md)
- 関連文書: [レビュー報告書](2026-08-19_code_review.md) / [IDPA解析](2026-08-19_idpa_analysis.md) / [統合設計書](2026-08-19_integration_design.md)

## 1. 何をしたか(要約)

IDPA リポジトリ(S2MFD のフォークに GA パラメタ推定を載せたもの、清水悠矢修論 / Shimizu & Hotta 2026, ApJ 996, 102 の実装)を、S2MFD 本体のサブパッケージ **`S2MFD.inference`** として統合した。同時に S2MFD 本体の重要バグを修正し、テスト基盤(pytest、90+ テスト)を新設した。IDPA リポジトリ自体は変更していない(アーカイブとして参照)。

## 2. 新しい使い方

### シミュレーション(従来どおり + 改善)
```python
import S2MFD
sim = S2MFD.run_simulation()               # Simulation オブジェクトが返るようになった
cfg = S2MFD.Cfg('parameters/hotta10.py')   # 絶対パスも渡せるようになった
cfg.rey = 1400; cfg.resolve()              # 基本量変更→派生量再計算(新設)
```

### パラメタ推定(新設)
```bash
python -m S2MFD.inference.cli --list-minima                     # 極小年の確認
python -m S2MFD.inference.cli --start-year 1944 --end-year 1996 \
    --pop 30 --generations 50 --seed 42                          # 論文設定
python -m S2MFD.inference.cli --interactive                      # 旧 IDPA.py 互換の対話モード
```
出力: `results/datas_{開始}_{終了}/data_{u0,s0}_{開始}_{終了}/` に `stage_result.json`(全結果)、`parameters.txt`(旧互換)、比較プロット、GA履歴、最良個体のスナップショット。
詳細は Sphinx ドキュメントの Parameter Inference の章(`doc/source/inference.rst`)。

### テスト
```bash
pip install -e ".[dev]"
pytest              # 通常テスト(~10秒)
pytest -m slow      # 長時間ベンチマーク
```
ゴールデンデータの再生成は `python tests/golden/generate_golden.py`(物理を意図的に変えたときのみ。worklog に記録すること)。

## 3. 旧 IDPA → 新モジュール対応表

| 旧(IDPA リポジトリ) | 新(S2MFD) |
|---|---|
| `IDPA.py`(対話ドライバ) | `S2MFD/inference/cli.py`(argparse。`--interactive` で対話互換) |
| `GA_for_u0.py` / `GA_for_s0.py`(98%重複) | `S2MFD/inference/genetic.py`(param_spec 引数の1実装) |
| `DefunctionProblem` + `run_defunction_simulation` | `S2MFD/inference/problem.py` の `DynamoProblem` |
| `judge`〜`judge5`(Simulation メソッド) | `S2MFD/inference/metrics.py`(純関数) |
| `make_time_series.py` + `detect_minimum.py` | `S2MFD/inference/observations.py` |
| `OBS_compare.py` | `S2MFD/inference/compare.py` |
| `make_graph.py` | (廃止。compare.py に吸収) |
| `initial_for_OBS` / `initial_for_LAST` | `Simulation.spin_up()` + `set_field()` |
| `defunction_main_loop` | `Simulation.run_window()` |
| `so0_time_dependent` / `uu0_time_dependent` / `uu0_known`(2重規約) | `cfg.uu0_of_time(t)` / `cfg.so0_of_time(t)`(単一規約) |
| `snumbers_energy`(7箇所にコピペの γ) | `S2MFD.tools.sunspot_proxy()` |
| `parameters/parameter_u0.py` / `parameter_s0.py` | `parameters/inference.py` + `problem.linear_plus_sines` |
| `obs_data/obs_data/SN_Yearly_interp.csv`(派生物) | `inference/data/SN_Yearly.csv`(原本)+ コード内挿 |
| `initial_data/{A,B}pht_saved.npy` | `inference/data/` にパッケージデータ化 |
| bisection 系 ~900行(旧世代手法) | **持ち込まず**(必要なら IDPA リポジトリ参照) |
| `IDPA/ana/` 14スクリプト | **持ち込まず**(ローカルパスのハードコード) |

## 4. 修正したバグ(主要なもの)

本体(詳細: レビュー報告書):
1. 出力タイムスタンプの1ステップずれ(保存場が time−dt の状態だった)
2. save/load でスカラーが0次元 ndarray 化する問題(NpzIO mixin で統一)
3. cont_flag 再開時の config.json 不整合の黙認(整合性チェック追加。デフォルト True は意図した仕様として維持)
4. パッケージング破損(非 editable install 不能)→ pyproject [project] に集約
5. 派生パラメタの読込時固定(cfg.resolve() 新設)
6. NaN 発散の無検知(check_finite 追加)
7. *_type 文字列スイッチの else なし(未知名は即 ValueError)

GA 層(詳細: IDPA 解析報告書 §4):
1. 並列評価の同一ディレクトリ書き込み競合 → 個体評価をディスクレス化して根絶
2. `GA_for_s0.py:50` genfromtxt 引数バグ(カスタムデータで Stage2 必ず失敗)→ 統合で消滅
3. 個体数<10 のトーナメントサイズ0クラッシュ → 3/7/5 固定+ガード
4. Stage1 の a0_s が Stage2 に渡らず捨てられる → stage_result.json で受け渡し
5. パラメタファイルへのソースコード書き換え → 廃止(JSON 受け渡し)
6. カスタムデータ指定時の年→行番号変換が既定 CSV 固定 → 指定データを使用
7. judge3 の obs/sim 逆転(period_mse で修正)、judge5 MAPE のゼロ割(eps ガード)
8. シードなし・print 洪水・KeyboardInterrupt の UnboundLocalError → 解消

## 5. 意図的に挙動を変えた点(結果の互換性)

| 変更 | 影響 |
|---|---|
| タイムスタンプ修正 | 各スナップショットの中身が旧実装の「1ステップ後」になる。**ラン終端の状態は旧実装とビット同等**(検証済み: 差 ~1e-14 は grid 座標の linspace 化による丸めのみ) |
| 時間依存パラメタの更新タイミング | ループ順序修正の帰結で旧実装から1ステップ(dt≈1.6日/更新間隔40日)ずれる |
| GA のエリート保存 | 固定スロット3 → ランダム1個体の置換 |
| GA のトーナメントサイズ | len//10, //4, //6 → 固定 3/7/5(個体数30では同値) |
| GA の乱数 | random モジュール → numpy Generator(数列は旧実装と別物。seed で再現可能に) |
| スピンアップの極小検出 | 打ち切り(max_extra、既定100年)を追加(旧実装は発振しない解で無限ループ) |
| 変異 | 既定は論文どおり非有界。`--clip-mutation` でクリップをオプション提供 |
| 最終ランの Lead_Time/ 保存 | 未移植(スピンアップのスナップショット保存。必要なら spin_up 後に手動 save) |

## 6. 検証結果

- **テスト**: 92+ passed(微分演算子・境界条件・Setup ゴールデン・小規模ラン回帰・リスタート・GA・観測処理・128×128 実格子での個体評価)。slow マーカーで αΩ 成長・potential BC 安定性。
- **パイプライン健全性**: twin experiment で「真値パラメタの適応度 = 0.900000」(理論最大値: 相関1・NMSE 0)を確認。シミュレーション→黒点数プロキシ→適応度の全経路が自己無撞着。
- **縮小版 twin experiment**(スピンアップ10年・窓22年・個体数12×8世代 ≈ 論文の評価回数の6%、seed=42): Step1 適応度 0.892/0.9・u0(t) 時系列誤差 13.5%、Step2 適応度 0.895/0.9・s0(t) 時系列誤差 14.1%。時系列として真値へ収束する傾向を確認(個々の sin 係数は縮退で大きくばらつく)。詳細は worklog の Phase 5。
- **注意**: 論文フルスケール(個体数30×50世代、80年スピンアップ、~50年窓)は1評価あたり約3万ステップ×4.8ms ≈ 2.3分、全体で8コア数時間規模。フルスケールの精度検証(論文表4.1: r=0.995, u0誤差2.87%)は未実施 → 残課題1。

## 7. 残課題

### すぐ着手できるもの
1. **フルスケール twin experiment**: 論文設定での精度再現確認(数時間の計算)。`results/` 一式が出る CLI で `--data` に合成データ CSV を渡せばよい。
2. **観測データでの実推定**: 1944–1996 等の区間で論文図4.5相当の再現。
3. GitHub Actions での CI(pytest 自動実行)。

### 論文5.1章の将来課題との対応
| 論文の課題 | 現状のコードでの足がかり |
|---|---|
| 推定区間の連続接続(前区間の終状態を初期条件に) | `Simulation.set_field()` + スナップショット読込で実装可能な構造になっている |
| 蝶形図の適応度への利用 | `run_window(on_output=...)` で任意の場の時系列を記録できる |
| u0・s0 の同時推定 | `DynamoProblem` の mode を拡張し param_spec を結合すれば試せる |
| 時間変化関数の改良 | `linear_plus_sines` を差し替え(TimeFunction を任意関数に) |
| 適応度の改良 | `metrics.py` に追加し `fitness_kind` で切替 |
| ダルトンミニマム | `--fitness period`(式4.2)を実装済み。精度向上は未着手 |

### コード側の残課題(レビュー報告書 §5)
- 移流スキームの風上化(GA が Pe>2 域に入ると格子振動。現状は NaN 検知でクラッシュは防げるが振動は検知しない)。
  なお探索範囲の 97.6% は Pe<2 の安全域にあることを確認済み
  (`doc/dev_records/verification/peclet_check.py`)
- potential BC の高次モード(n≈127)のエイリアシング
- ~~time_marching の njit 化などの高速化~~ → **2026-08-19 実施済み(14倍)**。
  詳細は [2026-08-19_optimization.md](2026-08-19_optimization.md)
- 境界条件のコーナーゴーストセル(margin>1 にする場合のみ)

## 9. 高速化 (2026-08-19, ブランチ `feature/speedup-time-marching`)

1 個体評価が **51.7 秒 → 3.6 秒 (14倍)**。物理量は変わらない
(黒点数時系列の相互相関 1.000000000000、適応度は小数10桁一致)。
詳細と検証結果は [2026-08-19_optimization.md](2026-08-19_optimization.md)。

使う上で知っておくこと:

- 既定は高速経路。参照実装とビット一致させたい場合は
  `cfg.exact_arithmetic = True`(約3倍遅い)。
- 高速経路は除算を逆数乗算に置き換えているため、参照実装との差は
  1 substep あたり倍精度 1 ULP 程度。散逸系なので積分しても増幅しない。
- 24 並列でも劣化 9% とほぼ線形にスケールする。個体数分のコアを与えれば
  1 世代 ≒ 1 個体の時間で回る。
- GA は親プロセスで `DynamoProblem.warmup()` を呼んでから fork するので、
  ワーカーごとの JIT コンパイル待ちは発生しない。

**計算時間の目安 (paris, EPYC 9354, 個体数30):**

| 作業 | 最適化前 | 最適化後 |
|---|---:|---:|
| 1 個体評価 | 51.7 秒 | 3.6 秒 |
| 1 区間 (u0+s0, 各50世代) | 約 4.4 時間 | 約 15 分 |
| 論文フル解析 (6区間) | 約 26 時間 | **約 1.5 時間** |

## 8. 歴史的メモ

- 削除した `S2MFD/scipy_test.py`(426行)は potential 境界条件のルジェンドル直交性・ナイキスト限界(lmax=128 問題)の検証試行だった。git 履歴(main 6cc3846 以前)から復元可能。修論付録5.2 の議論に対応。
- IDPA リポジトリの bisection 系コード(~900行)は GA 以前の旧世代推定手法。統合には持ち込んでいない。必要なら IDPA リポジトリ(`hottahd/IDPA`)を参照。
- 開発環境: リポジトリ直下 `.venv`(numpy 2.5 / numba 0.67 / Python 3.12)。旧 requirements のピン(numpy 1.24 等)は Python 3.12 非互換だったため撤廃し、pyproject の依存宣言に移行。
