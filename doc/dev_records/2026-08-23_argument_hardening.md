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
