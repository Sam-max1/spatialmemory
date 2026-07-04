# Memory Policy Rules — SMA

This policy defines the promotion matrix, eviction policies, privacy restrictions, and database safety requirements.

## 1. Promotion Matrix
- **Working Memory $\rightarrow$ Episodic**: Triggered on keyframe selection. Selection requires novelty index $>\tau_n$ or visual information gain $>\tau_v$.
- **Episodic $\rightarrow$ Long-Term**: A sighting is promoted to the scene graph only after $\ge 3$ observations from $\ge 2$ different view angles. Voxel structures are fused into LTM via confidence-weighted averaging.

## 2. Eviction and Decay Rules
- **Working Memory**: Fixed ring buffer size of 64 keyframes. FIFO eviction is applied unless a keyframe holds high-information metrics and is flagged for episodic promotion.
- **Episodic**: Retained at full fidelity for 90 days, then consolidated into session summaries.
- **Long-Term**: Voxel cells decay by confidence weight ($\theta = 0.15$) when not observed for $N = 20$ sessions, eventually being freed from RocksDB storage.

## 3. Privacy Scrubbing (Mandatory Gate)
- Face, screen, and document scrubbing must run in `metadata_enricher` before any frame is written to Episodic or Long-Term Memory.
- Storing unblurred keyframes or visual embeddings containing PII violates system safety and is prohibited.

## 4. Reversible Writes (Journaling)
- All writes to Long-Term Memory (pose optimization updates, voxel updates, loop-closures) must be journaled in the episodic log to support instant rollbacks if reflection checks indicate high residuals.
