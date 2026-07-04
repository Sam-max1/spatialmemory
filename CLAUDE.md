# CLAUDE.md — Spatial Memory Agent (SMA)

This document contains guidelines, environment configuration, command-line guides, and operational rules for Antigravity or any AI assistant working on the SMA project.

## Build and Run Commands
- **Install Package (Local Development)**: `pip install -e .[full]`
- **Run Flask UI Dashboard**: `python sma_app.py`
- **Run Tests**: `pytest`
- **Run Linters (Ruff)**: `ruff check .`
- **Run Type Checker (Mypy)**: `mypy .`

## Dual-Clock Rule (Critical Architecture)
- **Perception Clock**: Frame-rate bound (10-20 FPS, ≤50 ms latency target). **NO LLM CALLS, NO NETWORK HOPS, NO HEAVY GLOBAL OPTIMIZATIONS** are allowed on this thread. Must use local models (ONNX, PyTorch, FAISS) and in-memory structures.
- **Query Clock**: Query-rate bound (triggered by human/agent, ≤1.5s latency target). LLM processing and semantic parsing (e.g. Claude API) are allowed.

## Memory-Tier Structure
1. **Working Memory (WM)**: Active GCT frame slots, GPU-pinned tensors, 64-keyframe FIFO ring buffer, dynamic tracklets. Cleared on session end.
2. **Episodic Memory (EM)**: SQLite database storing logs, sightings, trajectory coordinates, loop-closure attempts, and session summaries.
3. **Long-Term Memory (LTM)**: LSM RocksDB-based voxel map (5 cm TSDF) and SQLite-based hierarchical scene graph. Persists across sessions and sites.

## Scale-Tier Semantics
Every metric observation must carry a scale-confidence tag:
- `METRIC_VERIFIED`: Calibration confirmed against IMU, stereo baseline, or fiducial wheel odometry.
- `METRIC_ESTIMATED`: Monocular scale estimated via network depth/prior.
- `RELATIVE`: Monocular scale without absolute references (unit-less navigation).
