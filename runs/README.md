# runs — Rempel 2006 再現ランの投入と測定

> **名前について**: 当初 `run_paris/` だった (paris というホストで走らせる
> ための一式だったため)。ホストに依存しないので 2026-08-25 に `runs/` へ
> 改名した。2026-08-24 までの `doc/dev_records/` は当時の名前のままである。
>
> **これらはリポジトリのスクリプトで、`pip install` されるパッケージには
> 含まれない。** 使うにはリポジトリを clone すること。パッケージ本体
> (`import S2MFD`) だけでも同じことはできる — スクリプトはその使い方の例。
>
> 手順の説明はドキュメントの
> [Rempel (2006) を再現する](https://is-isee.github.io/S2MFD/main/tutorial_rempel.html)
> にある。


## この環境での使い方 (堀田研)

`/scr/a000` は paris / boston / astana で共有だが `/home` は別なので、
python は `/scr/a000/c0234hotta/venv-paris/bin/python` を使い、投入時に
`cd /scr/a000/c0234hotta/Repository/S2MFD` と
`PYTHONPATH=/scr/a000/c0234hotta/Repository/S2MFD` が要る
(S2MFD は pip install されていない)。ssh の既定 cwd は HOME。

**投入先はそのつど決める。** ルール化せず、投入前に

    ssh <host> "uptime; ps -eo user:20,nlwp,etime,args --sort=-nlwp | head"

で load と**誰の何が走っているか**を見る (`ps` はユーザ名を切り詰めるので
`user:20` と幅を指定する)。astana は共有ファイルサーバでもあり、
ほかの解析が走っていることが多い。

## ドライバ

| ファイル | 用途 |
|---|---|
| `relax_scan.py` | 流体のみの緩和 (Λ 効果による差動回転の形成) |
| `dynamo7.py` | 緩和 -> 磁場投入 -> 非運動学的ダイナモ |
| `regrid.py` | 別解像度の状態を補間する |
| `table1.py` | 結果を Rempel 2006 表 1 と突き合わせる |

引数は `名前=値` 形式で cfg を上書きする。`init=path` で緩和済み状態から始める。
`S2MFD_PARFILE` でパラメタファイルを選ぶ (既定 `parameters/rempel06.py`)。
**論文どおりに走らせるなら `parameters/rempel06_paper.py`**
(北半球・0.65-0.985 RSUN・108x72・margin=2)。

## 投入スクリプト

| ファイル | 内容 |
|---|---|
| `launch_relax.sh` | 論文設定の緩和 40 年 (表 1 列 2: DR = 0.27) |
| `launch_dynamo.sh` | 表 1 列 3/5/7 と図 3 の運動学的参照解 |
| `launch_csscan.sh` | 人工拡散 `sld_cs_factor` への依存性 |
| `launch_resscan.sh` | 解像度への依存性 (収束性検証) |
| `chain.sh`, `start_chain.sh` | 緩和の完走を待ってダイナモを自動投入 |

## 並列度

**`S2MFD_PARALLEL` で制御する** (`OMP_NUM_THREADS` ではない。
`S2MFD/physics/_jit.py` の docstring 参照)。既定 0 は素の njit で逐次。

108x72 のような小さい格子では 2-3 スレッドで頭打ち。実測 (108x144, paris):

| S2MFD_PARALLEL | wall [ms/step] |
|---|---|
| 0 (素の njit) | 5.82 |
| 1 (parallel=True, 1 スレッド) | 4.70 |
| 2 | 4.29 |
| 3 | 4.56 |
| 4 | 4.07 |

**0 -> 1 が一番効く** (同じ 1 コアのまま 24% 速い。ParallelAccelerator の
ループ融合であってスレッドの効果ではない)。0/1/2/3/4 で結果はビット一致。

## 測定用 (使い捨て)

`bench_step.py`, `bench_cpu.py`, `bench*.sh`, `scaling.sh`, `repeat.sh`,
`serial.sh`, `probe.sh` は上の表を取るために使ったもの。
`bitcheck.py`/`bitcheck.sh` は `S2MFD_PARALLEL` の設定違いで結果が
ビット一致することの確認用。
