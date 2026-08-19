# S2MFD
Solar/Stellar Mean Field Dynamo code

軸対称2次元 (r, θ) の運動学的平均場(磁束輸送)ダイナモコード。
遺伝的アルゴリズムによるパラメタ推定機能 (`S2MFD.inference`) を含む。

- モデル・数値スキーム: Jouve et al. (2008) 準拠 + 圧縮項
- パラメタ推定手法: [Shimizu & Hotta (2026), ApJ 996, 102](https://doi.org/10.3847/1538-4357/ae2855)

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
