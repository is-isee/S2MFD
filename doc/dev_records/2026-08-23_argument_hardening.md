# 引数の取り違えを起こせなくする (2026-08-23)

引き継ぎ文書の未解決 3 番。`poloidal_mag` の引数取り違え (`grid.drr2` の
代わりに `grid.drr` を渡して B_theta が 2 倍) が **2 箇所で独立に**
起きていたので、構造的に直す。

## 1. 位置引数 20 個超のカーネルをキーワード呼び出しにした

`S2MFD/physics/dynamic.py` の Python レベルの呼び出し 17 箇所。実際の
位置引数の数は引き継ぎ文書の記載より多かった:

| カーネル | 位置引数 |
|---|---|
| `hydro.angular_momentum_rhs` | **40** |
| `hydro.momentum_rhs` | **31** |
| `hydro.entropy_rhs` | 26 |
| `artdif.sld_diffuse_meridional` | 19 |
| `hydro.viscous_meridional_rhs` | 19 |
| `artdif.sld_diffuse_*` | 16 |
| `hydro.cfl_dt` | 15 |
| `hydro.mass_rhs` | 14 |
| (ほか 9 本) | 9-12 |

njit の中からの呼び出しには手を触れていない (Python レベルだけ)。

### ビット一致を確認した

108x72 の緩和済み状態に磁場を入れて 50 ステップ進め、`om1 vrr vth ro1
se1 pr1` の全バイトを比較。**sha256 が一致** (1c4c19ab97789a9d...)。

### 速度への影響: 上限 0.6 パーセント

機械が混んでいる (load 50/64) ので壁時計の A/B は当てにならない
(実際 1 回目は「キーワードの方が 20 パーセント速い」という雑音が出た)。
代わりに**ディスパッチ負荷を単独で測って呼び出し回数を数えた**。

| 引数の数 | 位置 | キーワード | 差 |
|---|---|---|---|
| 26 | 1.696 us | 2.234 us | +0.538 us |
| 40 | 4.328 us | 5.607 us | +1.279 us |

(小さい配列で 5 回繰り返して最小値。カーネル本体の時間は潰してある。)

呼び出しは **20 回/step** (RK2 の 2 段 x カーネル 10 本)。全部が 40 引数
だとしても 0.026 ms/step で、108x72 の 4.0 ms/step に対し **0.64 パーセント**。
実際は 9-40 引数の混合なのでこれより小さい。**安全のために払う価値がある。**

### 副産物: 名前の食い違いが 1 つ見えた

`hydro.angular_momentum_rhs(open_bottom=self.om1_bottom_dirichlet, ...)`。
確認したところ `open_bottom` は「下部境界を粘性フラックスに開ける」意味で、
Rempel の Omega_1 = 0 剛体回転リザーバ境界と対応しているので**正しい**。
キーワードにしたことで対応が目に見えるようになった。

## 2. `poloidal_mag` を直接呼べなくした

`S2MFD.physics.poloidal_from_potential(aph, grid)` を追加し、本体・解析・
実行スクリプトの呼び出しを全部そちらに寄せた。格子を渡すので**引数を
選べない**。

- `S2MFD/physics/physics_core.py` (参照実装)
- `S2MFD/physics/dynamic.py` (`DynamicSolver.poloidal_from_potential` が委譲)
- `ana/ana_common.py`
- `run_paris/dynamo7.py`, `corner_impact.py`, `filter_dissipation.py`

`poloidal_mag` の docstring も直した (**引数名が `drr2_` なのに説明が
「drr : Radial grid spacing」になっていた**。これも取り違えを誘った)。

### 破れを検出するテストを入れた

`tests/test_physics_core.py::TestPoloidalMagArgument`

- `test_helper_takes_the_grid_so_the_argument_cannot_be_wrong`
  包みが `grid.drr2` を渡した場合とビット一致すること
- `test_production_code_does_not_call_the_raw_kernel`
  `S2MFD/`, `ana/`, `run_paris/` が `poloidal_mag` を直接呼んでいないこと
  (包み自身を除く)。**わざと違反を入れて落ちることを確認済み。**

テストは間違った引数をわざと渡すので対象外にしてある。

## やらなかったこと

`physics_core.py` の融合カーネル呼び出し (`kernel(Bph, Aph, dt, ...)`) は
末尾が `*factors` の可変長なので、Python の文法上キーワードに混ぜられない。
ここは触っていない。

---

# 続き: `ana/` のゴーストセル添字 (未解決 2 番)

引き継ぎ文書の未解決 2 番は「`ana/` は長期間 B_theta が 2 倍の図を出していた」
だったが、`ana/` を読み直したら**別のゴーストセル絡みの誤りが 2 種類**
残っていた。どちらも margin=1 では偶然正しく、**margin=2 (Rempel 設定) で
だけ壊れる**という見つけにくい形。

## (a) 表面を `Brrt[-2]` で取っていた

`grid.rr` はゴースト込みなので、最外の物理セルは `ixg - margin - 1`。

| パラメタ | margin | ixg | 最外物理 | `-2` |
|---|---|---|---|---|
| defaults / alpha_omega / hotta10 | 1 | 130 | 128 | 128 (一致) |
| **rempel06_paper** | **2** | **112** | **109** | **110 (ゴースト)** |

margin=1 の runs では正しかったので、長い間気づかれなかった。

## (b) `1+np.argmin(abs(grid.rr - 0.7*RSUN))` で 1 セルずれていた

`grid.rr` はゴースト込みなので `argmin` だけで正しい添字が出る。`+1` は
**1 セル外側**を指す。rempel06_paper で 0.7008 R のつもりが 0.7027 R、
hotta10 で 0.6984 R のつもりが 0.7016 R。

`butterfly_diagram.py` / `J08_test.py` / `sunspots_number.py` /
`npz_edit.py` の 4 本すべてに入っていた (`npz_edit.py` は 0.8 R)。

## 直したもの

`ana/ana_common.py` に添字ヘルパを追加し、4 本すべてを置き換えた:

    physical_slice(grid)        物理セルだけの (slice, slice)
    radial_index(grid, r)       r [cm] に最も近い物理セルの添字
    colat_index(grid, deg)      余緯度 [度] に最も近い物理セルの添字
    surface_index(grid)         最外の物理動径セル (margin=1 なら -2 と同じ)
    physical_colat_deg(grid)    物理セルの余緯度 [度] (プロットの縦軸用)

蝶形図の pcolormesh も、縦軸と데ータの両方を物理セルに限定した
(以前はゴーストの緯度まで描いていた)。B_phi と B_r を取った半径を
標準出力に出すようにしたので、ずれていれば図を見る前に気づく。

## 図が変わる範囲

- margin=1 の runs: **表面の位置は変わらない**が、`1+` を落とした分
  B_phi の半径が 1 セル内側になる (0.7016 R -> 0.6984 R)。
- margin=2 の runs: 表面がゴーストセルから最外物理セルに変わるので
  **B_r の図は変わる**。

## テスト

`tests/test_tools.py::TestAnaIndexHelpers` (6 本)

- `surface_index` が margin=1/2 のどちらでも物理セルであること
- margin=1 では従来の `[-2]` と一致すること (既存の図が変わらないこと)
- `radial_index` が物理セル内で最近傍であること
- **旧実装 `1+argmin` が 1 セル外側だったこと**を関係式として固定
- `colat_index` が物理セル内であること
