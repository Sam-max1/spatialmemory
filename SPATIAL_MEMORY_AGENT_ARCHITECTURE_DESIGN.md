# Spatial Memory Agent (SMA) — Memory-Based Agent Architecture Design

Version: v5 (final of 5 refinement iterations — see ITERATION_LOG.md for deltas per pass)
License target: Apache-2.0
Status: Design-complete, implementation-ready

---

## Project Summary

SMA is an embodied-AI spatial intelligence system that converts a streaming monocular RGB (optionally RGB-D/IMU) feed into a **persistent, queryable, self-improving spatial memory** — not a disposable point cloud. It wraps a feed-forward streaming reconstruction frontend (LingBot-Map-class Geometric Context Transformer) inside a full memory-augmented agent: three-tier memory with an explicit Memory Manager, retrieval-first spatial reasoning, and a reflection loop that measurably improves map quality, relocalization success, and query accuracy across sessions. Users are robotics teams, AR platforms, and inspection systems that need "where am I, what is around me, where did I see X, and how do I get back there" — across power cycles, not just within one run.

## Use Case Classification

Other — **Embodied spatial-intelligence agent** (perception + memory + spatial reasoning). Closest listed category: Knowledge management bot, where the "knowledge" is metric-semantic scene structure.

## Assumptions

Inferred from the project description (LingBot-Map / LingBot-Depth analysis); none stated by the user:

1. Primary sensor is one calibrated monocular RGB camera at ≥10 FPS, 518×378 or higher; RGB-D and IMU are optional plug-ins, not requirements.
2. Target deployment spans workstation GPU (RTX 4090-class) for full stack and embedded (Jetson Orin-class) for the perception+working-memory subset.
3. The perception frontend is an existing open feed-forward streaming model (LingBot-Map or VGGT/CUT3R-class) used as a **replaceable component**; SMA does not retrain a 3D foundation model from scratch in phase 1.
4. Scenes contain dynamic agents (people, vehicles, moved objects); the static-world assumption of the source projects is explicitly rejected.
5. Metric scale must be recoverable to ≤2% error when any auxiliary signal exists (IMU, known-baseline stereo, fiducial, wheel odometry); pure monocular runs are labeled scale-relative.
6. Maps persist across sessions per deployment site; multi-site tenancy is one long-term store per site ID.
7. Natural-language spatial queries ("where's the fire extinguisher on floor 2") are a first-class interface alongside programmatic APIs for planners.
8. Operators provide sparse feedback (correct/incorrect relocalization, object label fixes); no dense annotation budget exists.
9. Privacy constraint: faces, screens, and documents captured incidentally must never persist to long-term memory.
10. Latency ceiling: perception loop ≤50 ms/frame P95 on workstation GPU; spatial queries ≤1.5 s P95 end-to-end.
11. Storage budget per site: ≤20 GB long-term map after compaction for a ~10,000 m² facility.
12. Single-robot per session in v1; the map schema is multi-session-mergeable so multi-robot arrives without schema migration.
13. The LLM used for query understanding and spatial reasoning is API-hosted (Claude-class); embeddings run locally.
14. Network may be absent at runtime: every safety-critical path (perception, relocalization, obstacle geometry) must run fully offline; only NL query understanding may degrade to a rule-based parser.

## System Architecture Diagram (ASCII)

```
                                   ┌──────────────────────────────────────────────┐
                                   │              REFLECTION LOOP                 │
                                   │  drift monitor · map-quality evaluator ·     │
                                   │  object lifecycle tracker · feedback intake  │
                                   └───────▲──────────────────────────┬───────────┘
                                           │ signals                  │ validated updates
                                           │                          ▼
User / Robot ──► QUERY UNDERSTANDING ──► RETRIEVAL LAYER ──► MEMORY ARCHITECTURE ──► REASONING ENGINE ──► ACTION / RESPONSE
(camera feed,     intent: locate /        VPR (semantic) ·    Working: active recon    spatial QA ·          pose stream ·
 NL queries,      navigate / describe /   hybrid geo+sem ·    context (GCT KV state,   loop-closure          occupancy grid ·
 planner calls)   map / relocalize        multi-query ·       frame ring, tracklets)   arbitration ·         nav waypoints ·
                                          re-rank loop        Episodic: session        nav planning ·        NL answers w/
                                          candidates ·        trajectories, sightings, evidence synthesis    provenance
                                          scene-graph         relocalization events    over map + graph
                                          expansion           Long-Term: metric-
                                                              semantic map (voxel
                                                              hash + scene graph +
                                                              VPR index)
                                                              MEMORY MANAGER:
                                                              promote/evict/
                                                              summarize/route

              REASONING CYCLE (every frame batch and every query):
              01 Understand → 02 Retrieve → 03 Reason → 04 Act → 05 Reflect ──► Memory
```

## Layer-by-Layer Design

### Layer 1 — Query Understanding
**In Scope:** Yes.
**Components:**
- `intent_classifier.py`: Routes inputs into {frame-ingest, relocalize, locate-object, navigate-to, describe-region, map-audit, session-control}; frame-ingest bypasses the LLM entirely (hot path).
- `query_rewriter.py`: Resolves spatial deixis ("behind me", "the room I was just in") against current pose and working-memory trajectory before retrieval.
- `constraint_extractor.py`: Extracts floor/zone scope, recency bounds ("since yesterday"), and confidence requirements ("only if you're sure").
- `goal_state_detector.py`: Defines terminal answers per intent — e.g., locate-object terminates with (pose, confidence, last-seen timestamp, provenance frame IDs) or an explicit not-found with searched-region coverage.
**Memory Tier Involvement:** Reads working memory (current pose, recent trajectory) for deixis resolution; reads episodic for temporal references.
**Retrieval Strategies Active:** None directly; produces the retrieval plan.
**Key Design Decisions:**
- Dual-path input: sensor frames never touch the LLM parser — a rule-gated fast path preserves the 50 ms budget. LLM parsing applies only to NL queries.
- Offline fallback: a grammar-based parser covers the six intents when the LLM API is unreachable (Assumption 14).
**Dependencies:** Feeds Layer 2 retrieval plans; consumes Layer 3 working memory for context resolution.
**External Integrations:** Claude API (NL path only); local ONNX intent classifier for offline mode.

### Layer 2 — Retrieval Layer
**In Scope:** Yes. This is where loop closure, relocalization, and object search become *retrieval problems* — the core inversion relative to LingBot-Map, which buries retrieval inside attention.
**Components:**
- `retriever.py`: Unified interface over three indexes: VPR embedding index (place recognition), object-embedding index (open-vocabulary sightings), voxel/occupancy spatial index (geometric range queries).
- `query_expander.py`: For object queries, expands "fire extinguisher" → synonyms, CLIP-text variants, and container hypotheses ("wall mount", "red cabinet"); for relocalization, generates rotation-augmented query embeddings to handle revisits from novel headings.
- `reranker.py`: Two-stage — fast cosine top-K over VPR index, then geometric verification (feature matching + PnP inlier count) as the re-ranker. A loop-closure candidate that fails geometric verification is discarded regardless of embedding similarity. This is the anti-hallucination gate for the map.
- `context_expander.py`: On a scene-graph hit, pulls parent room node, sibling objects, and connecting-portal nodes so the reasoner answers "in the kitchen, on the counter left of the sink" instead of raw coordinates.
- `retrieval_router.py`: Routes by intent — relocalize→VPR+geometric verify; locate-object→object index+graph expansion; navigate→occupancy index; describe→graph traversal from pose.
**Memory Tier Involvement:** Reads all three tiers; retrieval results are raw context, never written back as memory (Principle 7).
**Retrieval Strategies Active:** All five (see Retrieval Strategy Matrix).
**Key Design Decisions:**
- Loop closure = semantic retrieval + geometric re-ranking, decoupled from the perception model's attention window. This removes LingBot-Map's 320-view horizon as a *correctness* bound: the GCT keeps only local context; global consistency comes from retrieval against the persistent index, at any distance, any session.
- Geometric verification threshold: ≥30 PnP inliers with reprojection error <2 px before a loop closure edge is accepted.
**Dependencies:** Consumes Layer 4 indexes; feeds Layer 5.
**External Integrations:** FAISS (IVF-PQ) for VPR/object indexes; SQLite R*-tree for spatial range queries.

### Layer 3 — Memory Architecture
**In Scope:** Yes — the centerpiece.

#### 3a. Working Memory
- Scope: Current session, current locality.
- Content: GCT anchor context + pose-reference window + trajectory tokens (the perception model's own three slots are *formalized as working memory*), a ring buffer of the last 64 keyframes with features, active dynamic-object tracklets, current pose covariance, in-flight query context.
- Implementation: Typed in-process store (`WorkingMemory` dataclass over pinned GPU/CPU tensors); no Redis on the hot path — a network hop violates the 50 ms budget. Snapshot-to-disk every 30 s for crash recovery.
- Eviction: Keyframe ring evicts FIFO with uncertainty-weighted exceptions (high-information frames marked for episodic promotion before eviction). Full clear at session end after summarization.

#### 3b. Episodic Memory
- Scope: Per-session records at one site.
- Content: Session summary (trajectory polyline, coverage polygon, distance, duration), object sighting events (embedding, pose, frame ID, timestamp, confidence), relocalization attempts with outcomes, drift-correction events, operator feedback, anomalies (blur storms, pose-collapse resets).
- Implementation: SQLite (WAL mode) per site; tables keyed by (session_id, timestamp); embeddings in sidecar FAISS index. Chosen over Postgres: embedded deployment, zero-ops, single-writer pattern fits.
- Write triggers: keyframe promotion events (streaming), sighting events (streaming), session-end summary (batch). All writes route through Memory Manager.

#### 3c. Long-Term Memory
- Scope: Persistent per-site spatial knowledge, session-independent.
- Content: (i) metric layer — voxel-hashed TSDF/occupancy at 5 cm resolution with per-voxel confidence and last-observed timestamp; (ii) semantic layer — scene graph: floors→rooms→surfaces→objects with open-vocabulary embeddings, inter-node spatial relations, portal edges; (iii) place layer — VPR keyframe index with poses; (iv) calibration layer — verified metric-scale factor and camera intrinsics history.
- Implementation: Voxel hash map in RocksDB (LSM suits append-heavy, range-compactable voxel writes); scene graph in SQLite with recursive-CTE traversal (Neo4j rejected: an embedded robot cannot carry a JVM graph server; graph size ~10⁴–10⁵ nodes per site is trivially within SQLite range); FAISS for the place layer.
- Update policy: Only via Memory Manager promotion after verification gates. Agent responses never write directly (Principle: no ad-hoc writes).

#### 3d. Memory Manager
- Control plane, `memory_manager.py`, methods `promote`, `evict`, `summarize`, `route`:
  - **Promote (working→episodic):** keyframe qualifies if (novelty: min embedding distance to existing place index > τ_n) OR (information gain: new voxel observations > τ_v) OR (event: loop closure, sighting, anomaly).
  - **Promote (episodic→long-term):** sighting becomes a scene-graph object node after ≥3 consistent observations across ≥2 viewpoints with reprojection agreement; voxel updates fuse with per-voxel confidence weighting; a place keyframe enters the VPR index only after geometric verification against neighbors.
  - **Evict:** working memory FIFO+exceptions (above); episodic sessions older than 90 days compress to summary-only; long-term voxels unobserved for N sessions decay confidence and free below threshold — this bounds map growth (see Memory Growth & Cost).
  - **Summarize:** session-end LLM/rule summary; region summaries ("storage room, 14 objects, last full coverage 2026-06-30") cached on graph nodes for fast describe-queries.
  - **Route:** maps retrieval requests to tier + index; enforces access tiers (Layer 8).
  - **Dynamic-object lifecycle:** objects observed absent from their recorded pose in ≥2 verified passes are demoted to "last known" status with the geometry cleared from occupancy — the mechanism LingBot-Map lacks entirely, and the reason its long maps rot (GitHub issue #26 class of failure: stale geometry accumulating as truth).

### Layer 4 — Knowledge Structures
**In Scope:** Yes.
**Components:**
- `chunker.py`: Spatial chunking, not text chunking — keyframe selection (temporal chunks), voxel blocks 8³ (metric chunks), room segmentation via portal detection (semantic chunks).
- `metadata_enricher.py`: Every keyframe/voxel-block/object carries: session_id, timestamp, pose, pose covariance, scale-confidence tier {metric-verified, metric-estimated, relative}, sensor config hash, privacy flags.
- `graph_builder.py`: Open-vocabulary detector (YOLO-World/OWLv2-class) + SAM-class masks → 3D-lifted object nodes; relation extraction (on/in/near/attached) from geometry; portal (door/opening) detection for room topology.
- `hierarchy_mapper.py`: site→building→floor→room→surface→object taxonomy, auto-assigned from graph topology, operator-correctable.
- `index_config/`: per-index configs — VPR embedding model + dim + IVF params; object index config; voxel resolution + truncation; graph schema version.
**Memory Tier Involvement:** Structures are the schema of episodic and long-term memory.
**Retrieval Strategies Active:** Defines what each strategy searches over.
**Key Design Decisions:**
- Scale-confidence tier on *every* metric datum: prevents silent mixing of scale-relative and metric-verified geometry — the failure LingBot-Map invites by leaving monocular scale implicit.
- Privacy flags computed at enrichment time (Layer 8 detector), before anything can persist.
**Dependencies:** Feeds Layers 2, 3; consumes raw perception output.
**External Integrations:** Open-vocab detector + segmenter (local, ONNX/TensorRT); embedding models local.

### Layer 5 — Reasoning Engine
**In Scope:** Yes.
**Components:**
- `reasoning_chain.py`: Spatial chain-of-thought for NL queries — grounds every claim in retrieved graph nodes/voxels/sightings; refuses to answer location questions from LLM parametric knowledge (Principle 1: a spatial answer without a retrieval provenance chain is a hallucinated map fact).
- `evidence_synthesizer.py`: Resolves conflicts — two sightings of the same object class at different poses → same object moved vs. two instances, decided by embedding similarity + temporal ordering + exclusivity constraints; conflicting voxel observations resolved by confidence-weighted fusion with recency bias.
- `decision_engine.py`: respond / retrieve-more (widen VPR K, relax filters) / act (emit waypoint, trigger re-observation of a stale region) / escalate (relocalization failed → request operator hint or start scale-relative fresh submap) / clarify (ambiguous referent: "two fire extinguishers known — floor 1 lobby or floor 2 east?").
- `tool_caller.py`: Tools with schemas — `plan_path(occupancy, start, goal)` (A*/hybrid-A* on occupancy), `reobserve(region_id)`, `merge_submaps(a,b,sim3)`, `export_map(format)` (PLY/GLB/occupancy PNG), `calibrate_scale(signal)`.
- `response_formatter.py`: NL answers with provenance (frame thumbnails, timestamps, confidence); JSON pose/waypoint schemas for planners; map exports.
- Loop-closure arbitration lives here: retrieval (Layer 2) proposes verified candidates; the reasoner decides whether to accept the pose-graph edge and trigger a lightweight pose-graph optimization (sparse, keyframe-poses-only — seconds, background thread), then Memory Manager writes corrected poses. This gives global consistency **without** per-frame optimization, preserving the feed-forward hot path.
**Memory Tier Involvement:** Reads all tiers via retrieval; writes nothing directly.
**Retrieval Strategies Active:** Consumer of all.
**Key Design Decisions:**
- Two clocks: perception loop (per-frame, no LLM, ≤50 ms) and query loop (per-request, LLM-backed, ≤1.5 s). One reasoning-cycle abstraction, two schedulers.
- Background pose-graph optimization is the *only* optimization in the system, bounded (keyframe poses only, ≤10⁴ nodes), asynchronous, and triggered solely by accepted loop closures — the middle path between LingBot-Map's "no optimization ever" (drift accumulates, windows compound error) and classical SLAM's "optimize everything" (kills real-time).
**Dependencies:** Layers 1–4; feeds Layer 6.
**External Integrations:** Claude API (query loop); GTSAM pose-graph solver (background thread, keyframe poses only).

**Reasoning loop (five mandatory steps):**
1. **Understand** — parse frame/query intent, resolve spatial references against pose (Layer 1).
2. **Retrieve** — VPR/object/occupancy retrieval per routed plan; geometric re-ranking (Layers 2+3).
3. **Reason** — synthesize evidence, arbitrate loop closures, resolve object identity, plan.
4. **Act** — emit pose/geometry, answer with provenance, send waypoint, or trigger re-observation.
5. **Reflect** — score outcome, emit drift/quality/lifecycle signals, route validated learnings to Memory Manager (Layer 6).

### Layer 6 — Reflection Loop
**In Scope:** Yes. This converts a mapping system into a self-improving one.
**Components:**
- `outcome_evaluator.py`: Per-frame: pose covariance trend, reprojection residuals, tracking-health score. Per-loop-closure: post-optimization residual. Per-query: retrieval hit verified geometrically? user accepted? Per-session: coverage delta, relocalization success rate.
- `learning_extractor.py`: Extracts — reliable-place statistics (which keyframes repeatedly verify → boost retrieval prior), unreliable-region flags (glass corridors where VPR fails → route multi-query with wider K), object-permanence priors (chairs move, extinguishers don't → per-class staleness half-life), scale-drift patterns.
- `memory_updater.py`: Sole write path for learnings → Memory Manager; every update carries evidence references and is reversible (journaled).
- `feedback_integrator.py`: Operator thumbs on relocalizations and answers, label corrections, "map is wrong here" pins → converted to confidence adjustments and re-observation tasks.
- `improvement_log.py`: Append-only ledger (what was learned, when, evidence, effect) — auditability of map evolution.
**Design principle applied:** every session emits ≥1 signal; the improvement metric is explicit: relocalization success rate and locate-query precision must be non-decreasing over rolling 10-session windows (regression trips an alert, Layer 7).
**Memory Tier Involvement:** Reads working+episodic; writes to episodic/long-term via Memory Manager only.

### Layer 7 — Observability & Evaluation
**In Scope:** Yes.
**Components:**
- `tracer.py`: Per-stage latency (understand/retrieve/reason/act/reflect), per-frame perception timing, GPU memory; OpenTelemetry export; LangSmith-class tracing for the LLM query path only.
- `eval_pipeline.py`: Offline eval on golden dataset: ATE/RPE vs. ground truth trajectories (TUM/ScanNet/Oxford-Spires splits + site-recorded sequences), relocalization recall@1 across session pairs, object locate precision/recall, occupancy IoU vs. surveyed floor plan.
- `online_monitor.py`: Live sampling — pose covariance, loop-closure acceptance rate, retrieval verification failure rate, memory growth rate vs. projection; alert thresholds.
- `golden_dataset.json`: ≥60 triples per site class: NL query / expected answer node / expected provenance frames, plus 12 trajectory sequences with GT poses and 8 cross-session relocalization pairs.
- `eval_metrics.py`: ATE, RPE, relocalization recall@1, locate-P/R, occupancy IoU, answer faithfulness (every claimed location backed by provenance), hallucinated-geometry rate (claims with zero voxel/graph support — target 0).
**Key Decision:** Faithfulness is checkable mechanically here (does provenance exist and geometrically verify), unlike text RAG — exploit that: no LLM-as-judge on the spatial-claims path.

### Layer 8 — Security & Safety
**In Scope:** Yes.
**Components:**
- `input_guard.py`: Visual-channel injection defenses — adversarial-patch/anomalous-texture detector on frames feeding VPR (a poster of another room is a spoofing vector for relocalization); NL prompt-injection filter on the query path; QR/text-in-scene never interpreted as instructions.
- `content_filter.py`: Hallucinated-geometry gate (any spatial claim without verifiable provenance is blocked, not softened); toxicity irrelevant, replaced by *safety-critical-output gate*: navigation waypoints must verify against current occupancy before emission.
- `output_filter.py`: Privacy scrubbing — faces, screens, documents blurred in stored keyframes and excluded from embeddings (detector runs in `metadata_enricher.py`, filter enforces at every export); map exports respect access tier.
- `access_controller.py`: Roles — robot-runtime (read map, write via MM only), operator (feedback, corrections, exports of own site), admin (cross-site, schema ops), guest-query (NL queries, no raw keyframe access). Enforced at Memory Manager route().
**Key Decision:** Privacy is enforced at *write time* (nothing sensitive persists), not export time — Assumption 9 made structural.

### Layer 9 — Infrastructure & Orchestration
**In Scope:** Yes.
**Components:** `main.py`, `config.py` (env-driven: model paths, site ID, sensor config, feature flags incl. `OFFLINE_MODE`), `agent_runner.py` (dual-clock scheduler running the five-step loop on both clocks, retry/backoff on query path, watchdog+state-snapshot recovery on perception path), `session_manager.py` (create/resume — resume = relocalize against long-term memory, the capability the source projects lack entirely), Dockerfile (app; CUDA base), Dockerfile.edge (Jetson, TensorRT engines, perception+WM+relocalize subset), `docker-compose.yml` (app + RocksDB volume + optional telemetry collector; no separate DB servers — everything embedded by design), `pyproject.toml`.
**Orchestration framework: Custom.** Justification: LangChain/LlamaIndex/AutoGen assume request-response text agents; a 20 Hz perception loop with pinned-memory tensors and a background pose-graph thread cannot live inside their executors. The Claude API is called directly on the query path with function-calling; that is the only "chain".
**Dependency discipline (fixing source-project brittleness):** perception model exported to ONNX/TensorRT where possible; FlashInfer optional with SDPA fallback *tested in CI*, not merely documented; Kaolin excluded from runtime (offline rendering only, separate extra); single `pip install sma[edge|full]`.

### Layer 10 — Agent Context & Rules (`.claude/`)
**In Scope:** Yes (mandatory).
**Components:**
- `CLAUDE.md`: architecture summary, memory-tier map, dual-clock rule ("never put an LLM or network call on the perception path"), scale-tier semantics.
- `AGENTS.md`: agent roles (perception runner, query agent, reflection worker), tool inventory with schemas, reasoning-loop configuration per clock.
- `.claude/rules/code-style.md`: typed Python 3.11, no dynamic tensor shapes on hot path, dataclasses for memory records, ruff+mypy gates.
- `.claude/rules/testing.md`: per-layer test requirements, golden-dataset update protocol (adding a sequence requires GT provenance), coverage ≥85% off-hot-path.
- `.claude/rules/memory-policy.md`: tier write matrix (who/what/when), promotion thresholds, privacy-flag handling, eviction/decay parameters, journal-and-reversibility requirement.

---

## Memory Architecture Detail

### Working Memory Design
Scope: session + spatial locality. Content: GCT streaming state (anchor/window/trajectory tokens), 64-keyframe ring with features and poses, dynamic tracklets, pose covariance, active query context. Implementation: in-process typed store, GPU-pinned tensors, 30 s crash snapshots. Eviction: FIFO ring; promotion check on every eviction candidate; hard clear + summarize at session end. **Explicit reframing:** LingBot-Map's three context slots *are* a working memory — the source project's real contribution — but it has no tiers above it; SMA supplies them.

### Episodic Memory Design
Scope: per-site session records. Content: session summaries, sightings, relocalization events, drift events, feedback, anomalies. Implementation: SQLite WAL + FAISS sidecar. Write triggers: streaming promotions + session-end batch, all via Memory Manager. Retention: full fidelity 90 days, then summary-only compression.

### Long-Term Memory Design
Scope: persistent per-site map. Content/implementation: RocksDB voxel-hash TSDF (5 cm, per-voxel confidence+timestamp), SQLite scene graph (floors/rooms/surfaces/objects/portals + relations + region summaries), FAISS VPR index, calibration records. Ingestion: promotion pipeline only — geometric verification for places, multi-view consistency for objects, confidence-weighted fusion for voxels. No direct writes from any responder.

### Memory Manager Logic
Promote: novelty/information-gain/event gates (working→episodic); multi-observation + geometric-verification gates (episodic→long-term). Evict: ring FIFO with promotion-check; 90-day episodic compression; voxel confidence decay for regions unobserved N=20 sessions, free below θ=0.15. Summarize: session and region summaries cached on graph nodes. Route: intent→tier/index mapping + role enforcement. Lifecycle: per-class staleness half-lives (learned by Layer 6) drive demotion of moved objects and clearing of stale occupancy — the structural fix for long-map rot.

## Knowledge Structure Design

| Structure | Content in this project | Implementation | Retrieval role |
|-----------|------------------------|----------------|----------------|
| Documents | Raw keyframes + session logs (source of truth, privacy-scrubbed) | Compressed keyframe store (JPEG-XL) + SQLite logs | Provenance; re-verification substrate |
| Chunks | Keyframes (temporal), 8³ voxel blocks (metric), room segments (semantic) | Keyframe selector; RocksDB voxel hash; portal-based segmentation | Unit of promotion, fusion, and range query |
| Knowledge Graph | site→floor→room→surface→object nodes; on/in/near/portal edges; open-vocab embeddings per node | SQLite + recursive CTE; embeddings in FAISS | Multi-hop spatial QA; context expansion; describe-queries |
| Hierarchies | Spatial taxonomy (site…object) with operator-correctable assignments | `hierarchy_mapper.py` over graph topology | Scoped retrieval ("floor 2 only"); drill-down |
| Tags & Metadata | pose+covariance, timestamps, session ID, scale-confidence tier, privacy flags, sensor hash, per-class staleness | `metadata_enricher.py`, columns on every record | Filtering, ranking, provenance, privacy enforcement |

## Retrieval Strategy Matrix

| Query Type | Primary Strategy | Secondary Strategy | Re-ranking | Context Expansion |
|------------|------------------|--------------------|------------|-------------------|
| Relocalization (session resume) | Semantic (VPR embedding) | Multi-query (rotation/illumination variants) | Yes — geometric verification (PnP inliers) | No |
| Loop-closure detection (streaming) | Semantic (VPR, throttled 1 Hz) | Hybrid (embedding + trajectory-proximity prior) | Yes — geometric verification | No |
| Locate-object (NL) | Hybrid (object embeddings + graph filters) | Multi-query (synonym/CLIP-text expansion) | Yes — recency × sighting-confidence | Yes — scene-graph parents/siblings |
| Navigate-to | Structured (occupancy R*-tree range) | Graph (portal topology for floor changes) | No | Yes — portal edges |
| Describe-region | Graph traversal from pose | Semantic (region summaries) | No | Yes — full subtree |
| Map-audit ("what's stale?") | Metadata filter (last-observed, confidence) | — | No | Yes — group by room |

Default-semantic-only is a design error here as everywhere: unverified embedding matches would write false loop closures into the pose graph — the single most destructive failure available to this system. Hence re-ranking by geometric verification is mandatory on every map-writing retrieval path.

## Reasoning Loop Configuration

- **Understand:** perception clock — frame validity, tracking health, mode (tracking/lost/relocalizing); query clock — intent, deixis resolution, constraints, goal state.
- **Retrieve:** perception clock — throttled VPR loop-closure probes, local voxel context; query clock — routed multi-index retrieval per matrix.
- **Reason:** perception clock — GCT feed-forward inference, tracklet update, closure arbitration; query clock — evidence synthesis, identity resolution, path planning, clarification decisions.
- **Act:** perception clock — pose+geometry emission, memory-promotion proposals; query clock — provenance-grounded answer / waypoint / tool call / escalation.
- **Reflect:** perception clock — covariance and residual signals; query clock — outcome scoring, feedback capture; both — signals to Layer 6, validated learnings to Memory Manager.

## Reflection Loop Design

Signals captured: pose-covariance trend, closure residuals, retrieval verification failures, query outcomes, operator feedback, coverage deltas, per-class object-movement events. Written to memory: reliable-place priors, unreliable-region flags, per-class staleness half-lives, confidence adjustments, re-observation tasks — all journaled and reversible via Memory Manager. Improvement metric: relocalization recall@1 and locate-precision non-decreasing over rolling 10-session windows; hallucinated-geometry rate pinned at 0. A deployment that is not measurably better at session N+10 than at session N is a defect, not a limitation.

## Design Principles Compliance

1. **Retrieve before guessing** — spatial claims require provenance chains; the hallucinated-geometry gate blocks (not softens) ungrounded claims; navigation verifies against live occupancy.
2. **Keep context focused and fresh** — working memory is a bounded ring + GCT slots; stale geometry is decayed and demoted by lifecycle rules, never silently trusted (direct fix for issue-#26-class rot).
3. **Store only what is useful** — novelty/information-gain gates on promotion; 3-observation rule for object nodes; 90-day episodic compression.
4. **Prefer structured knowledge** — scene graph + voxel hash + tagged keyframes over raw point clouds; the point cloud is an *export format*, not the memory.
5. **Reflect continuously and improve** — every session emits signals; explicit non-regression metric; improvement ledger.
6. **Rank relevance before reasoning** — geometric verification re-ranks every closure/relocalization candidate; recency×confidence re-ranks sightings. Unranked embedding hits never reach the pose graph.
7. **Separate memory from raw context** — retrieved candidates are context; only Memory-Manager-verified promotions become memory. LingBot-Map conflates these (its "trajectory memory" is unrecallable attention state); SMA's inversion of that conflation is the thesis of the project.

## Technology Stack

| Layer | Component | Technology | Justification |
|-------|-----------|------------|---------------|
| Perception | Streaming 3D frontend | LingBot-Map GCT (swappable; VGGT/CUT3R adapters) | Best open feed-forward streaming quality; SMA treats it as replaceable, its horizon limits are contained by the memory architecture |
| Perception | Depth prior (optional) | LingBot-Depth v0.5 via adapter when RGB-D present | SOTA completion on transparent/reflective; strictly optional to preserve monocular story |
| Retrieval | VPR/object vector index | FAISS IVF-PQ (local) | Embedded, offline-capable, 10⁵-scale indexes; a hosted vector DB violates Assumption 14 |
| Retrieval | Spatial index | SQLite R*-tree | Zero-ops range queries at site scale |
| Memory | Working memory backend | In-process typed store, GPU-pinned | 50 ms budget forbids network hops |
| Memory | Episodic store | SQLite (WAL) | Embedded, single-writer, crash-safe |
| Memory | Long-term metric layer | RocksDB voxel hash | LSM fits append-heavy voxel fusion; proven embedded |
| Memory | Long-term graph | SQLite recursive-CTE scene graph | 10⁴–10⁵ nodes; Neo4j server unjustifiable on-robot |
| Orchestration | Framework | Custom dual-clock runner | Real-time loop incompatible with LangChain/LlamaIndex/AutoGen executors |
| LLM | Reasoning model | Claude (API, function calling); grammar parser offline | Spatial QA quality; offline degradation path required |
| LLM | Embedding models | Open VPR model (MegaLoc/NetVLAD-class) + SigLIP/CLIP-class, ONNX/TensorRT local | Offline requirement; edge deployability |
| Optimization | Pose-graph solver | GTSAM (background thread) | Sparse keyframe-pose optimization in seconds |
| Safety | Detectors | ONNX face/screen/document detector; patch-anomaly detector | Write-time privacy; VPR anti-spoofing |

## Cross-Cutting Concerns

### Latency Budget
Perception clock (workstation): understand ≤2 ms, retrieve (closure probe, amortized) ≤8 ms, reason (GCT inference) ≤30 ms, act ≤3 ms, reflect ≤2 ms → ≤45 ms P50 / 50 ms P95 (≥20 FPS). Edge profile: 100 ms P95 at 10 FPS, half-resolution. Query clock: understand ≤300 ms, retrieve ≤200 ms, reason ≤700 ms, act ≤100 ms, reflect async → ≤1.3 s P50 / 1.5 s P95. Background pose-graph optimization: ≤5 s, never blocks either clock.

### Failure Modes & Fallbacks
- Tracking loss (blur/occlusion): mode→lost; relocalize via VPR against long-term memory; if unresolved in 10 s, open a fresh scale-relative submap and continue mapping; merge later on verified closure (structural replacement for LingBot-Map's pose collapse, which has no recovery).
- False loop closure: geometric gate (≥30 inliers) + post-optimization residual check; residual spike → edge reverted from journal.
- Scale drift: continuous auxiliary-signal cross-check when available; drift >2% → recalibration event, affected records demoted one scale tier.
- LLM API down: grammar parser handles the six intents; perception unaffected (Assumption 14).
- Storage pressure: Memory Manager forces decay-eviction and episodic compression before writes fail; alert at 80% budget.
- Crash: resume from 30 s working-memory snapshot + relocalization.
- Adversarial VPR spoof: patch detector flags frame; closure candidates from flagged frames require doubled inlier threshold.

### Memory Growth & Cost
Drivers: keyframes (~40 KB each scrubbed+compressed; ~2k/session-hour pre-gate, ~300 post-gate), voxels (bounded by site volume × 5 cm resolution, not by time — revisits fuse, don't append), graph (bounded by real objects), episodic rows (~10³/session, compressed at 90 days). Projection for 10,000 m² site, daily 2 h sessions: steady state ≈ 9–14 GB after decay-eviction, within the 20 GB budget; growth is asymptotic, not linear — the property LingBot-Map's per-frame KV accumulation structurally cannot have.

### Hallucination Risk Profile
Highest risk: (i) LLM inventing locations — mitigated by provenance-required answering + mechanical faithfulness check; (ii) false loop closures — geometric verification + residual monitoring + journaled reversibility; (iii) depth-completion inventing geometry on transparent surfaces (known LingBot-Depth failure surface) — completed-depth voxels carry a lower confidence prior and require multi-view confirmation; (iv) object misidentification — 3-observation rule + operator feedback path. Residual risk concentrates in single-visit regions, which are labeled low-confidence in every answer touching them.

## Open Questions

1. Multi-robot concurrent mapping: schema is merge-ready (Sim(3) submap merge tool exists) but conflict resolution policy for simultaneous writes to one site is deferred — requires CRDT-style voxel fusion or a coordinator; not needed for v1 single-robot scope.
2. Perception-frontend fine-tuning: whether to fine-tune the GCT on site-specific data (glass-heavy facilities) or rely on LingBot-Depth priors is deferred pending Phase 8 eval data.
3. Outdoor/GPS fusion: current scale calibration assumes indoor auxiliary signals; GNSS integration deferred.
4. Principle-6 tension acknowledged: navigate-to and describe-region skip re-ranking because their retrieval is structured (R*-tree, graph traversal), not similarity-based — re-ranking is defined as mandatory before *reasoning over similarity-retrieved candidates*, which is satisfied.
5. Legal review of keyframe retention policy per jurisdiction (GDPR-class) pending; write-time scrubbing is designed to make the stored artifact non-personal, but counsel sign-off is a deployment gate, not an architecture change.
