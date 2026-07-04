# SMA — Refinement Iteration Log (5 passes)

Each pass reviewed both deliverables against a fixed audit lens, applied deltas, and recorded them. v5 is the shipped state.

---

## Iteration 1 — Baseline draft (v1)
Lens: cover the full 10-layer template against the researched failure surface of LingBot-Map/Depth.
State produced:
- Six-stage pipeline mapped; three memory tiers + Memory Manager drafted; loop closure framed as retrieval.
- Defects carried out of v1 (found and fixed in later passes): loop closure ran as unverified embedding match; Neo4j and Redis assumed by template default; scale ambiguity unaddressed; single latency budget for both LLM queries and perception frames (incoherent — an LLM cannot sit in a 20 FPS loop); no dynamic-object story; map growth unbounded (same defect as the source project); no offline mode.

## Iteration 2 — Real-time & metric correctness (v2)
Lens: can this actually run on a robot at frame rate, with trustworthy metric output?
Deltas applied:
- Split the reasoning loop into a **dual-clock** design: perception clock (no LLM, no network, ≤50 ms P95) and query clock (LLM-backed, ≤1.5 s P95). Rewrote Layer 1 with a rule-gated frame fast path.
- Removed Redis from working memory (network hop violates the frame budget); replaced with in-process typed store + 30 s crash snapshots.
- Added **scale-confidence tiers** {metric-verified, metric-estimated, relative} stamped on every metric record; auxiliary-signal calibration (IMU/stereo/fiducial/odometry) with ≤2% drift target and demotion on drift detection.
- Added tracking-loss recovery: lost → VPR relocalization → fresh scale-relative submap + later verified Sim(3) merge (replaces the source project's unrecoverable pose collapse).
- Rewrote latency budget with per-step numbers for both clocks and an edge profile.

## Iteration 3 — Memory discipline & map rot (v3)
Lens: does the map stay correct and bounded over months? (Directly targets LingBot-Map issue-#26-class degradation and unbounded KV growth.)
Deltas applied:
- Loop closure and relocalization made **retrieval + mandatory geometric re-ranking** (≥30 PnP inliers @ <2 px); unverified embedding hits can never write pose-graph edges. Added post-optimization residual check with **journaled, reversible** edge writes.
- Added background, bounded pose-graph optimization (GTSAM, keyframe poses only, async) — the single optimization in the system; hot path stays feed-forward.
- Defined promotion gates numerically (novelty τ_n, info-gain τ_v, 3-observation object rule) and eviction/decay (episodic 90-day compression; voxel confidence decay N=20 sessions, free below θ=0.15). Growth becomes **asymptotic**: voxels bounded by site volume, not time; produced the 9–14 GB steady-state projection against the 20 GB budget.
- Added **dynamic-object lifecycle** (per-class staleness half-lives, learned in Layer 6; absence-verified demotion clears stale occupancy) — the structural fix for long-map rot.
- Replaced Neo4j with SQLite recursive-CTE scene graph and justified it (embedded robot, 10⁴–10⁵ nodes); RocksDB chosen for the voxel hash with LSM rationale.

## Iteration 4 — Safety, privacy, evaluation rigor (v4)
Lens: adversarial and regulatory reality; can claims be measured?
Deltas applied:
- Added visual-channel threat model: adversarial-patch VPR spoofing detector, doubled inlier threshold for flagged frames; scene text/QR never interpreted as instructions; NL injection filter confined to query path.
- Privacy moved from export-time to **write-time**: face/screen/document scrubbing inside `metadata_enricher.py` before any persistence; enforcement re-checked at every export path; jurisdictional sign-off listed as deployment gate (Open Questions #5).
- Replaced text-RAG faithfulness metrics with mechanically checkable ones: **hallucinated-geometry rate = 0** hard gate (spatial claims require provenance that geometrically verifies); waypoints verified against live occupancy before emission.
- Specified golden dataset concretely (≥60 query/answer/provenance triples, 12 GT trajectory sequences incl. Oxford-Spires split used by the source project for direct comparison, 8 cross-session relocalization pairs) and numeric eval targets (recall@1 ≥0.90, locate-P ≥0.90, occupancy IoU ≥0.85, ATE ≤ source baseline).
- Added the reflection non-regression tracker (rolling 10-session windows) with alerting — makes "self-improving" falsifiable.

## Iteration 5 — Deployability & consistency audit (v5, shipped)
Lens: dependency brittleness of the source stack; internal consistency of both documents.
Deltas applied:
- Dependency discipline: ONNX/TensorRT export path; FlashInfer optional with **SDPA fallback tested in CI** (the source project documents the fallback but doesn't gate on it); Kaolin excluded from runtime entirely; `pip install sma[edge|full]`; Dockerfile.edge (Jetson) running the perception+WM+relocalize subset.
- Added OFFLINE_MODE end-to-end (grammar parser for all six intents; every safety-critical path network-free) and made it an assumption (A14), a config flag, a healthcheck item, and a CI test.
- Added `test_perception_adapters.py` + conformance requirement so the LingBot-Map frontend is provably swappable (VGGT/CUT3R adapters) — contains vendor risk.
- Consistency fixes: unified pose-graph solver to GTSAM in both Layer-5 text and the tech-stack table; reconciled Principle-6 compliance with the structured-retrieval intents via Open Questions #4; aligned checklist thresholds with architecture numerics (τ, N, θ, inlier gate) so the plan carries no unscoped stubs.
- Added `docs/comparison.md` deliverable: measured deltas vs. LingBot-Map (drift over 10k frames, session resume, queryability, memory growth curve) — the superiority claim becomes an artifact, not a narrative.

---

## Residual known limitations (post-v5, deliberate)
- Multi-robot concurrency deferred (schema merge-ready; policy open — OQ#1).
- Outdoor/GNSS fusion deferred (OQ#3).
- Perception-frontend quality remains upstream-bounded; SMA contains but does not eliminate frontend failure on textureless/glass scenes without the optional depth prior.
