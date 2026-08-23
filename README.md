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

| 項目 | 精度 | 固定しているテスト |
|---|---|---|
| 質量・角運動量の保存 | machine precision (1e-20) | `tests/test_conservation.py` |
| エネルギー収支 (Rempel 2006 式 20-22) | 原論文が明記する 0.001 (288x192 で) | 同上 |
| 空間の収束次数 | 2.00 (一様・非一様とも) | 同上 |
| Rempel (2006) 表 1 の再現 | 周期 18.0-18.1 年 (論文 18)、磁場と交換項は 1 割以内 | `doc/dev_records/2026-08-23_table1_result.md` |

`pytest` で 213 本のテストが走る。詳細は
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

## Documentation

### URL
https://is-isee.github.io/S2MFD/main

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
