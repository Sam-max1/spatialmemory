# AGENTS.md — Internal Agent Roles and Schedules

The SMA system employs a distributed multi-agent hierarchy running on distinct clocks to coordinate perception, memory curation, spatial reasoning, and visual reflection.

## Agent Role Inventory

### 1. Perception Runner (`perception-runner`)
- **Objective**: Ingest raw frame streams, estimate poses, track dynamic obstacles, select keyframes, and scrub sensitive content.
- **Clock Execution**: Stream-driven (10–20 Hz, target ≤50 ms).
- **Tool Inventory**:
  - `novelty_selector`: Filters frames on embedding delta.
  - `privacy_scrubber`: Blurs faces/monitors using ONNX YOLO/Face detector.
  - `tracklet_updater`: Integrates spatial trajectories.

### 2. Query Agent (`query-agent`)
- **Objective**: Parse natural language user intents, orchestrate retriever routines, construct geometric verification pipelines, and synthesize spatial QA responses with absolute provenance links.
- **Clock Execution**: Request-driven (on-demand, target ≤1.5 s).
- **Tool Inventory**:
  - `intent_classifier`: Parses user queries to specific actions.
  - `spatial_path_planner`: Generates paths using A* search on occupancy grids.
  - `provenance_verifier`: Extracts keyframe witnesses for spatial claims.

### 3. Reflection Worker (`reflection-worker`)
- **Objective**: Audit map quality post-session, run pose-graph optimizations, decay stagnant long-term voxels, and update per-class object permanence priors.
- **Clock Execution**: Async/Batch (Triggered on session close or background idle, target ≤5 s).
- **Tool Inventory**:
  - `pose_graph_optimizer`: Formulates and solves GTSAM pose optimization.
  - `decay_scheduler`: Prunes unobserved voxels ($N=20, \theta=0.15$).
  - `permanence_tuner`: Estimates object lifetime half-lives.
