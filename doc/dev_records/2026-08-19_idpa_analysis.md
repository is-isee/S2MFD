# IDPA 解析報告書(S2MFD への統合前調査)

- 日付: 2026-08-19
- 対象: `/scr/a000/c0234hotta/Repository/IDPA`(remote: `hottahd/IDPA`)HEAD `8d3b6f3`
- 参考: 清水悠矢 修士論文(2025年度)/ Shimizu & Hotta 2026, ApJ 996, 102(DOI: 10.3847/1538-4357/ae2855)
- 関連文書: [本体レビュー](2026-08-19_code_review.md) / [統合設計書](2026-08-19_integration_design.md)

## 1. リポジトリの素性

- IDPA は S2MFD の**git 履歴を共有しないコピー**。root commit `d69160b`(2025-12-05)は upstream `is-isee/S2MFD` とマージベースを持たない。→ **git merge/rebase による統合は不可能**。コードは新規パッチとして再実装・移植する。
- 内部パッケージ名は `S2MFD` のまま(`setup.py:5`, `pyproject.toml:8`)。upstream と**同時に pip install できない**。
- 「IDPA」という名前は修論・論文には登場しない(リポジトリ名のみ)。

## 2. 動作の全体像

```
IDPA.py(対話式ドライバ、input() プロンプト)
  ├─ データ選択(既定 'OBS' = SILSO 年平均黒点数の40日内挿)
  ├─ detect_minimum で極小年を提示 → 推定区間 [開始年, 終了年](≦60年)を入力
  ├─ 個体数・最大世代数・目標適応度を入力
  ├─ Stage 1: GA_for_u0 — u0 の時間変化5係数 + 定数 a0_s の6パラメタを推定
  ├─ ★ parameter_s0.py をソースコード書き換えで更新(Stage1 の結果を埋め込む)
  ├─ Stage 2: GA_for_s0 — s0 の時間変化5係数を推定(u0 は Stage1 の結果で固定)
  └─ OBS_compare — 黒点数比較プロット・数値結果の出力
```

- 時間変化の表現(修論式2.10): 線形項 + sin級数(lmax=4、l=3 は除外)→ 各パラメタ5係数。
- 適応度(式3.1): `F = 0.9·r − 0.1·NMSE`(黒点数時系列の相関 − 正規化二乗誤差、40日サンプリング)。
- 個体評価: 各個体につき Jouve+2008 の固定場(`initial_data/{A,B}pht_saved.npy`)から80年+極小期到達までスピンアップ(`initial_for_OBS`)→ 観測窓を積分(`defunction_main_loop`)→ 黒点数プロキシ(γ·B_φ²(0.7R☉, θ=75°), γ=5.8653520852)で比較。
- GA 設定: SBX交叉(η=2、80%)、ガウス変異(σ=0.1|x|、30%、非有界)、ASPトーナメント選択(適応度変化率と多様性で圧力切替)、エリート保存、`ProcessPoolExecutor` による個体並列評価。

## 3. upstream との共有ファイル差分

| ファイル | 差分 | 統合時の扱い |
|---|---|---|
| `simulation.py` | 191行→1470行。生きている追加は `initial_for_OBS`/`initial_for_LAST`/`defunction_main_loop`/`judge`〜`judge5`/`snumbers_energy` と save() 拡張・時間依存フックの約250行。**残り~900行は GA 経路から呼ばれない旧世代の二分法(bisection)コード** | 生きている分を汎用APIに再設計して移植。bisection 系は持ち込まない |
| `data.py` | `np.load(..., allow_pickle=True)` 1行 | 持ち込まない(save の追加キーをプレーン配列にすれば不要) |
| `__init__.py` | GA モジュールの import 追加 | inference サブパッケージの公開に置換 |
| `parameters/alpha_omega*.py` | `datadir` 1行 | 持ち込まない |
| `main_functions.py` | 空白のみ | 差分なし扱い |
| その他(grid, legendre, cfg, physics/, tools/) | **バイト一致** | — |

## 4. 発見したバグ・問題(統合時の対処)

| # | 問題 | 場所 | 対処 |
|---|---|---|---|
| 1 | 並列評価で全個体が同一 output_dir に書き込む競合(grid.npz, data.*.npz, config.json, P_sunspot.png)。最後に書いたプロセスの結果が残る | `GA_for_u0.py:994-1019` ほか | 個体ごとに `gen{g}/ind{i}/` を割当 |
| 2 | `genfromtxt` に位置引数3つ(datadir が dtype に化ける)→ カスタムデータは Stage2 で必ず失敗 | `GA_for_s0.py:50` | GA 2ファイル統合で消滅 |
| 3 | 個体数<10 でトーナメントサイズ `len//10`=0 → IndexError。コメントの意図は k=3/7/5 固定 | `GA_for_u0.py:293-325` | k=3/7/5 固定 + `max(2, min(k, len))` ガード |
| 4 | カスタムデータ指定時も年→行番号変換が常に `SN_Yearly_interp.csv` を読む | `IDPA.py:97`, `OBS_compare.py:50` | 指定データを使用 |
| 5 | Stage1 で推定した `a0_s` が Stage2 に渡らず捨てられる | `IDPA.py:116` vs `126-133` | JSON 受け渡しで解消 |
| 6 | パラメタファイルへのソースコード書き換え(前回実行の残骸が `parameter_s0.py:9-14` にコミット済み。prefix 一致で誤置換の危険、CWD 依存) | `IDPA.py:113-157` | 廃止 → JSON + cfg 属性 |
| 7 | `so0_time_dependent`/`uu0_time_dependent` が2つの非互換シグネチャで hasattr スニッフィングされ、`parameter_s0.py` を通常実行すると TypeError | `simulation.py:107-112` vs `337-348` | 単一コールバック規約 `uu0_of_time(t)` に統一 |
| 8 | ガウス変異が非有界(範囲逸脱自由、σ∝\|x\| で0付近凍結) | `GA_for_u0.py:481-491` | 既定は論文どおり維持、クリップ/σ下限をオプション化【ユーザー決定】 |
| 9 | RNG シードなし(再現不能)、`random` モジュール名の import 隠蔽 | 全体 | numpy Generator + `--seed` |
| 10 | KeyboardInterrupt ハンドラが初世代評価中の割込で UnboundLocalError | `GA_for_u0.py:769-795` | 修正 |
| 11 | print 洪水(並列ワーカー×選択/交叉/変異/judge 毎)。README の未完了項目「余計な出力をしないようにする」に対応 | 全体 | logging 化 |
| 12 | `judge3` の obs/sim 逆転(コード内 TODO で自認、現在未使用)、`judge5` MAPE のゼロ割(未使用) | `simulation.py:604-667` | metrics 移設時に修正 |
| 13 | CWD 依存のハードコードパス多数(obs_data/, initial_data/, PNG 出力先) | 各所 | 引数/cfg/パッケージデータへ |
| 14 | `_exec_prot_crossover` が存在しないキー 'B','C' を参照(選ばれると KeyError) | `GA_for_u0.py:432-443` | 持ち込まない |
| 15 | 評価で `sd`(judge2)`pd`(judge3)`MAPE`(judge5)を計算して捨てている(無駄コスト) | `GA_for_u0.py:994-1019` | 使う指標のみ計算 |
| 16 | GA_for_u0.py / GA_for_s0.py が98%コピー(~1050行重複、既にドリフトしてバグ#2が片方のみ) | 両ファイル | param_spec 引数で1本化 |

## 5. データファイル

- `obs_data/obs_data/SN_Yearly_interp.csv`: SILSO 年平均(1723-2024)を40日間隔に線形内挿した**派生物**(`make_time_series.py` で生成)。→ 統合では原本 `SN_Yearly.csv` + 生成関数をリポジトリに置き、内挿はコードで実行。monthly/13months CSV は未使用のため持ち込まない。
- `initial_data/{Apht,Bpht}_saved.npy`: スピンアップ用の Jouve+2008 平衡場(各135KB、128×128)。→ パッケージデータ化(`importlib.resources` で解決)。
- `IDPA/ana/`(14スクリプト): 全てローカルパスのハードコードで、`ana/OBS_compare.py` は `S2MFD/OBS_compare.py` の古い複製。→ 持ち込まない。

## 6. 論文どおり保つべき挙動(検証基準)

- GA: 個体数30・最大50世代・SBX(η_CO=2, 80%)・ガウス変異(σ=0.1|x|, 30%)・ASP(ε_fit=0.005, ε_div=0.10)・エリート保存。
- 2段階推定: Step1 u0(s0定数)→ Step2 s0(u0既知)。同時推定は局所解のため不採用(論文4章)。
- 初期集団範囲: u0系 [500,900]/[−150,150]、s0系 [40,65]/[−15,15]。
- twin experiment No.1 の到達水準: Step2 後 r=0.995、u0誤差2.87%、s0誤差5.25%(論文表4.1)。
