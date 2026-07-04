# Testing Rules — SMA

The SMA codebase relies on rigorous validation to prevent geometric drift, loop-closure regressions, and LLM reasoning hallucinations.

## 1. Coverage Targets
- Maintain at least **85% code coverage** for all off-hot-path components (Reasoning, Reflection, Safety, Retrieval).
- Focus tests on boundary conditions (e.g. tracking failures, SQLite connection timeouts, corrupted RocksDB logs).

## 2. Layer-Specific Testing Requirements
- **Perception/Retrieval**: Test that place recognition (`retriever.py`) achieves $\ge 90\%$ recall@1 on golden cross-session pairs.
- **Reranker**: Ensure the geometric verification gate rejects known-false candidate frames.
- **Memory**: Verify working memory ring eviction, promotion conditions, and database transaction rollback integrity.
- **Reasoning**: Enforce that spatial QA blocks parametric guessing when there is no retrieval provenance.

## 3. Golden Dataset Integrity
- Never modify tests in a way that weakens the golden-dataset provenance check.
- Adding new trajectory sequences requires providing matching ground truth poses.
