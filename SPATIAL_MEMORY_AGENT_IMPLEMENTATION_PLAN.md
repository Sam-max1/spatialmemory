# Spatial Memory Agent (SMA) — Implementation Plan

Version: v5 (final of 5 refinement iterations — see ITERATION_LOG.md)

## Checklist Legend
- [ ] Not started
- [~] In progress
- [x] Complete

## Phase 0 — Project Scaffold & Agent Context
- [ ] Initialize repository structure: `sma/` package with `perception/ memory/ retrieval/ reasoning/ reflection/ safety/ infra/` subpackages
- [ ] Configure `pyproject.toml`: Python 3.11, extras `[edge]` (TensorRT, no Kaolin/FlashInfer) and `[full]` (workstation)
- [ ] Set up `docker-compose.yml`: app service + RocksDB volume + optional OTel collector (no external DB servers — embedded stores only)
- [ ] Create `.env.example`: SITE_ID, MODEL_PATHS, SENSOR_CONFIG, ANTHROPIC_API_KEY, OFFLINE_MODE, STORAGE_BUDGET_GB, PRIVACY_MODE
- [ ] Write `CLAUDE.md`: architecture summary, dual-clock rule (no LLM/network on perception path), scale-tier semantics, memory-tier map
- [ ] Write `AGENTS.md`: perception-runner / query-agent / reflection-worker roles, tool schemas, per-clock loop config
- [ ] Write `.claude/rules/code-style.md`: typed 3.11, static tensor shapes on hot path, dataclass memory records, ruff+mypy gates
- [ ] Write `.claude/rules/testing.md`: per-layer requirements, golden-dataset GT-provenance protocol, ≥85% coverage off hot path
- [ ] Write `.claude/rules/memory-policy.md`: tier write matrix, promotion thresholds (τ_n, τ_v, 3-obs rule, 30-inlier gate), decay params (N=20, θ=0.15), privacy-flag handling, journal/reversibility rule
- [ ] Write `README.md`: quickstart (webcam demo → persistent map → NL query → session resume)

## Phase 1 — Knowledge Structures & Data Pipeline
- [ ] `chunker.py`: keyframe selector (novelty + info-gain), 8³ voxel-block partitioner, portal-based room segmenter
- [ ] `metadata_enricher.py`: pose+covariance, timestamps, session ID, scale-confidence tier, privacy flags (face/screen/document detector inline), sensor hash on every record
- [ ] `graph_builder.py`: open-vocab detection (OWLv2-class ONNX) + mask lifting → object nodes; on/in/near relation extraction from geometry; portal detection for room topology
- [ ] `hierarchy_mapper.py`: site→floor→room→surface→object auto-assignment from portal topology; operator-correction API
- [ ] `index_config/`: vpr.yaml (embedding model, dim, IVF-PQ params), objects.yaml, voxel.yaml (5 cm, truncation, decay), graph_schema.sql (versioned)
- [ ] Data ingestion script: recorded-video → frames → perception → enrich → promote → indexes end-to-end, for bootstrapping sites from existing footage

## Phase 2 — Memory Architecture
- [ ] `working_memory.py`: typed store wrapping GCT anchor/window/trajectory state + 64-keyframe ring + tracklets + covariance; 30 s crash snapshots; FIFO-with-promotion-check eviction
- [ ] `episodic_memory.py`: SQLite WAL schema (sessions, sightings, relocalizations, drift_events, feedback, anomalies) + FAISS sidecar; 90-day compression job
- [ ] `long_term_memory.py`: RocksDB voxel-hash TSDF with confidence-weighted fusion; SQLite scene graph with recursive-CTE traversal; FAISS place index; calibration records
- [ ] `memory_manager.py`: promote (novelty/info-gain/event gates; 3-obs + geometric-verification gates), evict (ring, 90-day compression, voxel decay N=20/θ=0.15), summarize (session + region summaries), route (intent→tier/index + role enforcement), journaled reversible writes, dynamic-object lifecycle demotion

## Phase 3 — Retrieval Layer
- [ ] `retriever.py`: unified interface over VPR index, object index, R*-tree occupancy index
- [ ] `query_expander.py`: object synonym/CLIP-text expansion; rotation/illumination-augmented VPR query variants
- [ ] `reranker.py`: geometric verification stage — feature matching + PnP, ≥30 inliers @ <2 px gate; recency×confidence ranking for sightings
- [ ] `context_expander.py`: scene-graph parent/sibling/portal expansion for grounded spatial answers
- [ ] `retrieval_router.py`: intent→strategy routing per Retrieval Strategy Matrix; 1 Hz streaming closure-probe throttle
- [ ] `prompts/retrieval_templates.py`: five patterns instantiated spatially (basic locate, region summary, time-aware "latest sighting", question-focused, expansion "related objects/rooms")

## Phase 4 — Query Understanding
- [ ] `intent_classifier.py`: {frame-ingest, relocalize, locate-object, navigate-to, describe-region, map-audit, session-control}; rule-gated fast path for frames, LLM path for NL
- [ ] `query_rewriter.py`: spatial deixis resolution ("behind me", "the room I just left") against pose + trajectory
- [ ] `constraint_extractor.py`: floor/zone scope, recency bounds, confidence requirements
- [ ] `goal_state_detector.py`: terminal-answer schema per intent incl. explicit not-found with searched-coverage
- [ ] Conversational state tracking integration with working memory (multi-turn referent carry-over)
- [ ] Offline grammar parser covering all six intents (OFFLINE_MODE)

## Phase 5 — Reasoning Engine
- [ ] `reasoning_chain.py`: provenance-required spatial CoT; refuses parametric-knowledge location answers
- [ ] `evidence_synthesizer.py`: object-identity resolution (moved vs. duplicate), confidence-weighted voxel conflict resolution with recency bias
- [ ] `decision_engine.py`: respond / retrieve-more / act / escalate (relocalization failure → operator hint or fresh scale-relative submap) / clarify (ambiguous referents)
- [ ] `tool_caller.py`: plan_path (A* on occupancy), reobserve(region), merge_submaps(Sim3), export_map(PLY/GLB/occupancy), calibrate_scale — all schema-enforced
- [ ] `response_formatter.py`: NL answers with provenance thumbnails/timestamps/confidence; JSON pose/waypoint schemas; map exports
- [ ] Loop-closure arbitration + background GTSAM pose-graph optimization (keyframe poses only, async, journaled edge writes)
- [ ] `agent_runner.py`: dual-clock scheduler (perception ≤50 ms P95; query ≤1.5 s P95) with watchdog, snapshot recovery, query-path retry/backoff

## Phase 6 — Reflection Loop
- [ ] `outcome_evaluator.py`: covariance trends, closure residuals, retrieval-verification failures, query outcomes, session coverage/relocalization stats
- [ ] `learning_extractor.py`: reliable-place priors, unreliable-region flags (glass corridors), per-class staleness half-lives, scale-drift patterns
- [ ] `memory_updater.py`: sole learning write path via Memory Manager; evidence-referenced, journaled, reversible
- [ ] `feedback_integrator.py`: operator thumbs on relocalizations/answers, label corrections, "map wrong here" pins → confidence adjustments + re-observation tasks
- [ ] `improvement_log.py`: append-only ledger (learning, timestamp, evidence, effect)
- [ ] Non-regression tracker: relocalization recall@1 and locate-precision over rolling 10-session windows, alert on regression

## Phase 7 — Security & Safety
- [ ] `input_guard.py`: adversarial-patch/anomalous-texture detector on VPR-bound frames; NL prompt-injection filter; scene text/QR never executed as instructions
- [ ] `content_filter.py`: hallucinated-geometry hard gate (zero-provenance spatial claims blocked); waypoint-vs-live-occupancy verification before emission
- [ ] `output_filter.py`: write-time face/screen/document scrubbing enforcement on keyframes, embeddings, and every export path
- [ ] `access_controller.py`: robot-runtime / operator / admin / guest-query roles enforced at memory_manager.route()
- [ ] Spoofed-relocalization defense: doubled inlier threshold for flagged frames

## Phase 8 — Observability & Evaluation
- [ ] `tracer.py`: per-stage + per-frame latency, GPU memory, OTel export; LLM-path tracing
- [ ] `golden_dataset.json`: ≥60 query/answer/provenance triples, 12 GT trajectory sequences (TUM/ScanNet/Oxford-Spires + site recordings), 8 cross-session relocalization pairs
- [ ] `eval_pipeline.py`: offline ATE/RPE, relocalization recall@1, locate-P/R, occupancy IoU vs. surveyed plans, faithfulness (mechanical provenance check)
- [ ] `online_monitor.py`: covariance, closure-acceptance rate, verification-failure rate, memory-growth-vs-projection alerts
- [ ] `eval_metrics.py`: targets — ATE ≤ LingBot-Map baseline on identical sequences; relocalization recall@1 ≥0.90; locate-P ≥0.90/R ≥0.80; hallucinated-geometry rate = 0; occupancy IoU ≥0.85

## Phase 9 — Infrastructure & Entry
- [ ] `main.py`: entry point; modes {live, replay, query-only, ingest}
- [ ] `config.py`: env-driven config incl. OFFLINE_MODE, storage budget, privacy mode
- [ ] `session_manager.py`: create / resume (resume = VPR relocalization against long-term memory) / terminate (summarize + clear WM)
- [ ] Dockerfile (app, CUDA)
- [ ] Dockerfile.edge (Jetson: TensorRT engines, perception+WM+relocalize subset; no Kaolin, no FlashInfer)
- [ ] `docker-compose.yml`: finalized (app + volumes + optional OTel)
- [ ] `healthcheck.py`: model load, index integrity, storage headroom, camera probe, LLM reachability (non-fatal)
- [ ] `seed.py`: bootstrap a site from recorded footage via Phase-1 ingestion
- [ ] `migrate.py`: graph-schema and index-config version migration
- [ ] CI: SDPA-fallback test matrix (FlashInfer absent must pass), ONNX export test, edge-profile latency smoke test

## Phase 10 — Tests
- [ ] `test_retrieval.py`: relocalization recall on golden pairs per strategy; verification-gate rejects known-false candidates; expansion correctness
- [ ] `test_memory.py`: WM ring + snapshot recovery; promotion gates (novelty/info-gain/3-obs); voxel decay math; journal reversibility; episodic compression
- [ ] `test_reasoning.py`: provenance-required answering (parametric answers blocked), identity resolution cases, path-planning validity, closure arbitration + residual revert
- [ ] `test_reflection.py`: signal emission per session ≥1, staleness half-life learning, non-regression tracker triggers
- [ ] `test_security.py`: patch-spoof relocalization probes fail, NL injection corpus blocked, privacy scrubbing verified on stored keyframes and all exports, role matrix enforced
- [ ] `test_integration.py`: end-to-end — live map → kill process → resume → relocalize → NL locate with provenance → navigate; dual-clock latency budgets asserted
- [ ] `test_perception_adapters.py`: LingBot-Map adapter + VGGT/CUT3R adapter conformance (frontend swappability proven)

## Phase 11 — Documentation
- [ ] `docs/architecture.md`: system overview, layer-by-layer, dual-clock model
- [ ] `docs/memory-design.md`: tier policies, promotion/eviction/decay math, growth projections, journal format
- [ ] `docs/api-reference.md`: query API, planner API (pose/waypoint schemas), tool schemas, export formats
- [ ] `docs/deployment.md`: workstation + Jetson profiles, env vars, site bootstrap, index initialization, privacy configuration
- [ ] `docs/comparison.md`: measured deltas vs. LingBot-Map baseline (drift over 10k frames, resume capability, query capability, memory growth curves)

## Dependency Map
| Component | Depends On | Blocks |
|-----------|------------|--------|
| `memory_manager.py` | working/episodic/long_term stores | `reasoning_chain.py`, reflection loop, `session_manager.py` |
| `retriever.py` | `index_config/`, `chunker.py`, `metadata_enricher.py` | `reranker.py`, `reasoning_chain.py` |
| `reranker.py` (geometric gate) | `retriever.py` | loop-closure arbitration, pose-graph writes |
| `graph_builder.py` | perception adapter, `metadata_enricher.py` | `context_expander.py`, locate/describe intents |
| `reasoning_chain.py` | `retriever.py`, `memory_manager.py` | `agent_runner.py` |
| `agent_runner.py` | Layers 1–6 components | `main.py`, `test_integration.py` |
| `outcome_evaluator.py` | `golden_dataset.json`, `agent_runner.py` | `memory_updater.py`, non-regression tracker |
| Privacy scrubbing (`metadata_enricher` + `output_filter`) | detectors | any persistence to episodic/long-term (hard gate) |
| Perception adapters | LingBot-Map / VGGT checkpoints | everything downstream (first runnable milestone) |

## Completion Gates
| Gate | Criterion | Phases Required |
|------|-----------|-----------------|
| Perception baseline | LingBot-Map adapter reproduces published ATE on Oxford-Spires split; SDPA fallback passes CI | 1, 9 |
| Retrieval baseline | Relocalization recall@1 ≥0.90 on golden cross-session pairs; verification gate FPR ≤1% | 1, 3 |
| Memory integrity | WM cleared+summarized at session end; promotions verified; journal revert proven; growth curve asymptotic in 20-session soak | 2 |
| Reasoning correctness | Zero hallucinated-geometry on golden set; closure arbitration reverts injected false edges; dual-clock budgets met | 4, 5 |
| Reflection active | ≥1 signal per session; measured recall@1 improvement across 10-session soak; regression alert fires on injected degradation | 6 |
| Security clearance | All patch-spoof and NL-injection probes fail as expected; zero unscrubbed sensitive pixels in stored/exported artifacts | 7 |
| Eval baseline | ATE ≤ LingBot-Map baseline on identical sequences; locate-P ≥0.90; occupancy IoU ≥0.85 | 8 |
| Production readiness | Healthcheck green; 10k-frame + resume soak within latency and 20 GB budgets on both profiles; docs complete | 9, 10, 11 |
