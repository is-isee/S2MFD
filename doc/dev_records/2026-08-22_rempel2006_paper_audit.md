# Rempel (2006) 原論文との照合と、見つかった設定ミスの修正 (2026-08-22)

原論文 (`~/reference/Rempel_2006_*.pdf`, `Rempel_2005_*.pdf`) を直接読み、
実装と一項目ずつ突き合わせた記録。**転記メモ
(`2026-08-19_rempel_equations.md`) だけを頼りにしていたために見落として
いた設定ミスが 2 件**見つかった。

## 経緯

ダイナモ 4 本 (n985a-d) の結果で、トーショナル振動が論文の 4.7 nHz に対し
44-65 nHz と一桁大きかった。ユーザーから「ローレンツ力の 8pi のファクターを
間違えていないか」という指摘があり、そこから照合を始めた。

**8pi は誤りではなかった。** 既知の解析磁場を入れて
`angular_momentum_rhs` が出す Maxwell 応力ソースを独立計算と突き合わせた
ところ、比 1.00000000、相対残差 2.0e-15 で機械精度一致した
(4pi 版と一致。8pi なら 0.5 になる)。子午面の (rot B) x B / 4pi、
磁気エネルギー B^2/(8pi)、SLD のアルヴェン速度 B/sqrt(4pi rho) も正しい。

一桁の食い違いの正体は次の 2 つだった。

1. 比較先の取り違え。4.7 nHz は alpha_0 = 0.125 m/s の値で、
   alpha_0 = 0.25 m/s なら表 1 は 11.5 nHz。
2. `om_bar` を磁場投入時の値で固定していたため、差動回転の**永年的な減少**を
   振動振幅として計上していた (下記 修正 3)。

なお「差動回転が磁場に削られるのは異常」という当初の見立ては**誤り**だった。
表 1 は alpha_0 とともに (Omega_eq - Omega_pole)/Omega_0 が
0.27 -> 0.21 -> 0.15 -> 0.10 と下がることを示しており、これは論文自身の結果。
むしろ我々の減り方の方が弱かった。

## 修正 1: alpha クエンチングは運動学的ランだけのもの

**根拠 (原論文):**

- §2.2 末尾 (運動学的参照解, 図 3): "We use for the alpha effect an amplitude
  of alpha_0 = 0.125 m/s and **include alpha quenching with a quenching field
  strength of 1 T (10 kG)**."
- §3.1 冒頭 (ローレンツ力フィードバックあり): "Since Lorentz force feedback
  introduces enough nonlinearity to saturate the dynamo, **it is not necessary
  to include alpha quenching** as typically done in kinematic models."
- 図 4 キャプション: "Dynamo solution with Lorentz force feedback **and no
  alpha quenching**."

表 1 の列 3-9 (本文の主結果) はすべてクエンチングなし。ところが実装は
`physics_core.py` で `alpha_fac = b_src/(1 + (b_src/beq)**2)` を
**無条件に**掛けていた。B_eq = 1 T で磁場が頭打ちになるため、実測の
max|B_phi| が 0.72-1.03 T と、表 1 の 1.2-1.4 T に届かなかった。

さらに転記メモは B_eq = 1 T を「式 10-11、§4.3 のみ」と正しく記録して
いた。式 10-11 は **nu_t と kappa_t のクエンチング**で、§4.3 (図 12) だけで
使われる**代替**のフィードバック機構。実装がこれを持っていないのは正しい。

**修正:** `cfg.alpha_quenching` を追加 (既定 True = 従来互換)。
`rempel06.py` は `alpha_quenching = False` (論文の主結果に合わせる)。
`dynamo7.py` は `kinematic` フラグに連動させる。

## 修正 2: Lambda 効果と alpha 効果の r_max が領域上端に固定されていた

Lambda 効果 (2005 式 33) の `tanh((r_max - r)/d)` と alpha 効果
(2006 式 17) の `max[0, 1 - (r - r_max)^2/d_alpha^2]` は、どちらも論文の
領域上端 r_max = 0.985 RSUN を基準にしている。実装はこれを `grid.rrmax` に
結びつけていたため、数値安定性のために領域を 0.96 RSUN に切り詰めると
**駆動の位置まで 0.025 RSUN 内側に動いていた**。論文 §2.2 は
"confines the poloidal source term above r = 0.935 R_sun" と明記しており、
0.96 領域では 0.91 RSUN になってしまう。

この影響で、8/21 に走らせた緩和ラン 3 本は互いに比較できない設定になって
いた (r985s/r985t は 0.985 + 伸縮格子、res216x144 は 0.96 + 一様)。
「解像度を上げると DR が 0.27 に寄る」という当時の結論は成立しない。

**修正:** `cfg.r_max` を追加 (既定は `grid.rrmax` = 従来互換)。
ただし r_max を領域上端からずらすと Lambda フラックスが上部境界で
ゼロにならず、Rempel 2005 §2.5 の "we require a vanishing angular momentum
flux at the top boundary" が破れるので、ずれていれば警告を出す。
**論文を再現するなら領域上端そのものを 0.985 RSUN にすべき**
(2 点集中格子で安定に走ることは r985s/r985t で実証済み)。

## 修正 3: トーショナル振動の基準は時間平均

表 1 の量は max(Omega - Omega_bar) で、Omega_bar は時間平均。
`dynamo7.py` は磁場投入時の Omega で固定していたため、ローレンツ力による
差動回転の永年的な減少がそのまま振幅として乗っていた。

**修正:** 時定数 1 サイクル (18 年、`om_bar_tau_yr` で変更可) の
指数移動平均に変更。

保存済みの履歴からの後追い補正も可能で、移動平均を引き直すと
n985a 10.3 -> 2.4 nHz、n985b 44.2 -> 9.6 nHz となり、表 1 の
4.7 / 11.5 nHz と同じ桁に収まる。

## 修正 4: 北半球のみを解く

**根拠 (原論文 §2.2):** "We restrict our simulations to one hemisphere and
impose the dipole symmetry through our equatorial boundary condition."
"The boundary condition is A = B_Phi = 0 at the pole and
**dA/dtheta = B_Phi = 0 at the equator**, which selects the dipole symmetry."
2005 §2.6: 領域は "from equator to pole", 解像度は "108 grid points in radius
and 72 grid points in latitude"。

走行中だった 4 本の B_phi の赤道パリティを測ると P = -1.0000 +- 0.0000 が
全時刻で成立していた。種磁場 sin(2 theta) が厳密に反対称で演算子が
パリティを混ぜないため、**全球計算は同じものを二度計算していた**。

**修正:**

- `boundary_condition`: 上端が赤道なら A_phi の鏡像符号を +1 に
  (B_phi は -1 のまま)。thmax から自動判定、`cfg.equator_top_bc` で上書き可。
- 流体側の `_mirror_th` は変更不要。極の正則性条件と赤道の対称性条件は
  符号が一致する (ro1, v_r, Omega_1, s_1 が +1、v_theta が -1)。
- `zero_boundary_faces_th` に `top_is_pole` を追加。**赤道は壁ではない**ので
  拡散フラックスをゼロにしてはいけない。反対称量 (v_theta, B_phi) の
  勾配は赤道でゼロではなく、全球計算では両半球がここで運動量と磁束を
  やり取りしている。5 つの hydro カーネルと 5 つの artdif カーネルに
  引数を通した (既定 True = 従来互換)。
- `EnergyBudget` の方位角因子: 全球なら 2pi、半球なら 4pi
  (2006 式 34 と表 1 の注 "we compute from that the energy conversion for
  the entire sphere")。`artificial_dissipation` の診断も同様。
- **`margin >= 2` が必要。** SLD のリミタ `sld_flux_th` は境界面の 2 セル先を
  参照するので、赤道のゴーストが 1 層だと片側差分に落ちる。margin=1 だと
  dq_mt が全球と 9e-5 ずれ、margin=2 で厳密に一致する。半球かつ margin<2
  なら警告を出す。

**検証:** `TestHemisphereEquivalence` で

- 右辺 5 成分すべてが全球の北半球とビット一致 (margin=2)
- 1 ステップ後は 1 ULP (4e-17) 以内
- 40 ステップ後は 1e-9 以内 (ダイナモの成長率で丸めが増幅されるため)

## 未解決

- alpha_0 = 0.25 の 2 本で Q_L^Omega/Q_Lambda が 0.44 に対し表 1 は 0.149。
  ただし 60 年のうち 36 年時点の値で、論文は複数サイクル平均。
  修正後に取り直す必要がある。
- 領域上端を 0.985 RSUN に戻すかどうか。`rempel06.py` の既定は数値安定性の
  ため 0.96 のまま。論文値で走らせるなら 2 点集中格子を併用する。

## 反省

転記メモ (`2026-08-19_rempel_equations.md`) は式そのものは正確だったが、
「alpha 効果 (式 16-19) にクエンチング項がない」ことを**書いていない**
だけでは、実装側が既存の運動学的ダイナモの機構を引き継いだことに
気づけなかった。**論文にあたるのを後回しにしたのが判断ミス**だった。
仕様書を作ったら、実装との照合は原論文に対して行うこと。
