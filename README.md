# S2MFD
Solar/Stellar Mean Field Dynamo code

軸対称2次元 (r, θ) の平均場(磁束輸送)ダイナモコード。**2 つのモード**を持つ。

| モード | 解くもの | 出典 |
|---|---|---|
| **運動学的** (既定) | 誘導方程式のみ。差動回転と子午面流は外から与える | Jouve et al. (2008) 準拠 + 圧縮項 |
| **動力学** (`dynamics='full'`) | 運動量・角運動量・連続・エントロピーの式も解き、**磁場が流れに反作用する** | Rempel (2005, 2006) |

遺伝的アルゴリズムによるパラメタ推定機能 (`S2MFD.inference`) を含む。

- パラメタ推定手法: [Shimizu & Hotta (2026), ApJ 996, 102](https://doi.org/10.3847/1538-4357/ae2855)

## 検証されていること

| 項目 | 精度 | 根拠 |
|---|---|---|
| 質量・角運動量の保存 | machine precision (1e-20) | `tests/test_conservation.py` |
| 磁場のエネルギー収支 (式 22) | Q_Λ 比で 1e-5 〜 2e-4 | `doc/dev_records/2026-08-23_energy_budget_residual.md` |
| 流れのエネルギー収支 (式 20-21) | 288x192 で 0.0012 (c_s=0.30) 〜 0.0024 (c_s=0.02) | 同上 |
| 空間の収束次数 | 2.00 (一様・非一様とも) | `tests/test_conservation.py` |
| **Rempel (2006) 表 1**: 周期 | 18.0-18.1 年 / 論文 18 (比 1.00-1.01) | `doc/dev_records/2026-08-25_table1_final.md` |
| **同**: エネルギー交換項 5 つ | 論文比 0.93-1.10 | 同上 |
| **同**: トーショナル振動 | 論文比 0.95-1.26 | 同上 |
| **同**: 磁場 (max B_φ, max B_r, Ē_B) | 論文比 0.83-1.12 | 同上 |
| 差動回転の収束極限 | DR_∞ = 0.266 ± 0.003 / 論文 0.27 | `doc/dev_records/2026-08-25_analysis_audit.md` |

差動回転の絶対値は論文比 1.10-1.15 だが、これは**論文の格子 (108x72) が
未収束**なことの持ち込みで、自分の運動学的参照解で規格化すると 144x96 で
0.999 / 1.004 になる。**この実装の付加価値は、論文の数値が未収束であることを
定量化した点にある。**

`pytest` で 222 本のテストが走る。詳細は
[動力学モードの章](https://is-isee.github.io/S2MFD/main/dynamic.html)。

---
## Installation

Currently S2MFD is under development. Install directly from GitHub:

```bash
$ git clone https://github.com/is-isee/S2MFD.git
$ cd S2MFD
$ pip install -e ".[dev]"   # 開発用 (pytest 込み)。利用のみなら pip install .
```

## Quick Start

You can run a simulation with the default parameters in `S2MFD/parameters/defaults.py`:
```python
import S2MFD
sim = S2MFD.run_simulation()
```

## Parameter Inference (GA)

黒点数観測から子午面流振幅 u0(t) と α効果振幅 s0(t) の時間変化を2段階推定する:

```bash
$ python -m S2MFD.inference.cli --list-minima            # 推定窓の候補 (極小年) を確認
$ python -m S2MFD.inference.cli --start-year 1944 --end-year 1996 \
      --pop 30 --generations 50 --seed 42
$ python -m S2MFD.inference.cli --interactive            # 対話モード
```

詳細はドキュメントの Parameter Inference の章を参照。

## Tests

```bash
$ pytest              # 通常テスト
$ pytest -m slow      # 長時間ベンチマーク
```

## Development records

開発経緯・レビュー記録・引き継ぎ資料は `doc/dev_records/` にある。
入口は **`handover_2026-08-25.md`** (最新の引き継ぎ)。

`runs/` は Rempel (2006) 再現ランの投入・解析スクリプト一式。
**リポジトリのスクリプトで、`pip install` されるパッケージには含まれない。**

## Documentation

### URL
https://is-isee.github.io/S2MFD/main

コードを読むための Python の前提 (`class` から積み上げる) は
[コードを読むための Python](https://is-isee.github.io/S2MFD/main/reading_the_code.html)
の章にある。

### Sphinx

ドキュメントは GitHub Actions (`.github/workflows/docs.yml`) が main への push /
タグ作成時に自動ビルドして GitHub Pages に公開する。ローカルでビルドを試す場合:

```bash
$ pip install -e ".[docs]"
$ sphinx-build -b html doc/source doc/build/html
```

## CI

GitHub Actions (`.github/workflows/test.yml`) が push / PR ごとに
pytest を実行する (Python 3.10 / 3.12)。
