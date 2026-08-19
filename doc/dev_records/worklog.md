# 作業ログ(IDPA 統合プロジェクト)

新しいエントリを上に追記する。各エントリは「何を・なぜ・どう検証したか」を記録する。

---

## 2026-08-19

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
