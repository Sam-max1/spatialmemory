# Code Style Rules — SMA

All code written in the SMA codebase must adhere to the following principles to maintain performance constraints and system reliability.

## 1. Type Enforcement
- Every function signature **must** include type hints.
- Type annotations are required on dataclasses and stateful variables.
- Run `mypy .` before committing code to ensure strict static checking passes.

## 2. No Dynamic Tensor Shapes on Hot Paths
- Avoid resizing PyTorch/ONNX arrays dynamically inside the perception loop.
- Pre-allocate frames, tracks, and voxel buffers where possible.
- Pinned GPU memory must be reused via buffer recycling.

## 3. Dataclasses for Memory Records
- Use structured, typed `dataclasses` (rather than plain dictionaries or lists) for all keyframe, sighting, pose, and query objects.
- Dataclass models should be serializeable to JSON/SQLite formats.

## 4. Linting
- Code must pass `ruff check .` with zero errors.
- Keep line lengths under 100 characters.
