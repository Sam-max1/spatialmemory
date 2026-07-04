import time
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from sma.infra.config import Config
from sma.memory.working_memory import WorkingMemory, KeyframeRecord
from sma.memory.episodic_memory import EpisodicMemory
from sma.memory.long_term_memory import LongTermMemory
from sma.memory.memory_manager import MemoryManager
from sma.perception.chunker import Chunker
from sma.perception.metadata_enricher import MetadataEnricher
from sma.perception.graph_builder import GraphBuilder
from sma.retrieval.retriever import Retriever
from sma.retrieval.query_expander import QueryExpander
from sma.retrieval.reranker import Reranker
from sma.retrieval.context_expander import ContextExpander
from sma.retrieval.retrieval_router import RetrievalRouter
from sma.reasoning.intent_classifier import IntentClassifier
from sma.reasoning.query_rewriter import QueryRewriter
from sma.reasoning.constraint_extractor import ConstraintExtractor
from sma.reasoning.goal_state_detector import GoalStateDetector
from sma.reasoning.reasoning_chain import ReasoningChain
from sma.reasoning.evidence_synthesizer import EvidenceSynthesizer
from sma.reasoning.decision_engine import DecisionEngine
from sma.reasoning.tool_caller import ToolCaller

class AgentRunner:
    def __init__(self, config: Config):
        self.config = config
        
        # Instantiate tiers
        self.wm = WorkingMemory()
        self.em = EpisodicMemory(config.SQLITE_EPISODIC_PATH)
        self.ltm = LongTermMemory(config.SQLITE_SCENE_GRAPH_PATH, config.ROCKSDB_VOXEL_PATH)
        self.mm = MemoryManager(self.wm, self.em, self.ltm, config.STORAGE_BUDGET_GB)

        # Instantiate perception components
        self.chunker = Chunker()
        self.enricher = MetadataEnricher(config.SENSOR_CONFIG)
        self.graph_builder = GraphBuilder()

        # Instantiate retrieval components
        self.retriever = Retriever(self.ltm, self.em)
        self.expander = QueryExpander()
        self.reranker = Reranker()
        self.context_expander = ContextExpander(self.ltm)
        self.router = RetrievalRouter(self.retriever, self.expander)

        # Instantiate reasoning components
        self.intent_classifier = IntentClassifier()
        self.query_rewriter = QueryRewriter()
        self.constraint_extractor = ConstraintExtractor()
        self.goal_detector = GoalStateDetector()
        self.reasoner = ReasoningChain(config.OFFLINE_MODE)
        self.synthesizer = EvidenceSynthesizer()
        self.decision_engine = DecisionEngine()
        self.tool_caller = ToolCaller()

        self.current_session_id = f"session_{int(time.time())}"
        self.em.start_session(self.current_session_id, time.time())

        # Mock occupancy grid for path planning simulations
        self.occupancy_grid = np.zeros((50, 50))
        # Inject some obstacles
        self.occupancy_grid[10:15, 10:35] = 1
        self.occupancy_grid[25:35, 15:20] = 1

    def run_perception_step(self, image_data: bytes, pose: Tuple[float, float, float, float]) -> Dict[str, Any]:
        """
        Perception loop (target latency <=50ms). NO heavy LLM or network requests.
        Selects keyframes, updates working memory pose & tracklets, and promotes to EM.
        """
        start_time = time.time()
        
        pose_np = np.array(pose)
        cov_np = np.eye(3) * 0.02 # simulated tracking covariance
        
        # Ingest state updates
        self.wm.update_pose(pose_np[:3], cov_np)
        
        # Process visual embeddings (mock feature generation)
        frame_emb = np.sin(pose_np[0] * np.arange(128))
        
        # Voxel updates (simulate voxel block updates based on robot motion)
        grid_coords = (int(pose[0]/0.05), int(pose[1]/0.05), int(pose[2]/0.05))
        self.ltm.set_voxel(grid_coords, 0.5, 0.9, time.time())
        
        is_kf, kf_metrics = self.chunker.evaluate_keyframe(
            frame_emb, 
            current_voxels=len(self.ltm.voxel_cache), 
            historical_voxels=max(1, len(self.ltm.voxel_cache)-2)
        )
        
        evicted_kf = None
        if is_kf:
            # Enrich frame metadata & check privacy flags
            enriched = self.enricher.enrich_frame(
                image_data, pose_np, cov_np, 
                scale_confidence="METRIC_ESTIMATED", 
                session_id=self.current_session_id
            )
            
            # Apply privacy scrubbing simulation if flags triggered
            if self.config.PRIVACY_MODE and any(enriched["privacy_flags"].values()):
                image_data = b"SCRUBBED_" + image_data[:10]
            
            # Insert into working memory
            kf_record = KeyframeRecord(
                frame_id=f"kf_{int(time.time() * 1000)}",
                timestamp=enriched["timestamp"],
                image_data=image_data,
                embedding=frame_emb,
                pose=pose_np,
                pose_covariance=cov_np,
                scale_confidence=enriched["scale_confidence"],
                privacy_flags=enriched["privacy_flags"],
                novelty_score=kf_metrics["novelty"],
                info_gain=kf_metrics["info_gain"]
            )
            evicted_kf = self.wm.add_keyframe(kf_record)
            
            # Write keyframe and track logs to episodic SQLite
            self.mm.promote_working_to_episodic(kf_record)

        self.ltm.save_voxels()
        latency = (time.time() - start_time) * 1000.0
        
        return {
            "is_keyframe": is_kf,
            "latency_ms": latency,
            "evicted_keyframe": evicted_kf is not None,
            "working_memory_keyframes": len(self.wm.state.keyframe_ring)
        }

    def run_query_step(self, user_query: str) -> Dict[str, Any]:
        """
        Query loop (target latency <=1.5s). Runs intent classification,
        deixis re-writing, memory retrieval, CoT reasoning, and formatting.
        """
        start_time = time.time()
        
        # Deixis resolution
        pose_4d = (self.wm.state.current_pose[0], self.wm.state.current_pose[1], 0.0, 0.0)
        trajectory = self.em.get_session_trajectory(self.current_session_id)
        rewritten_query = self.query_rewriter.rewrite_deixis(user_query, pose_4d, trajectory)
        
        # Intent classification
        intent, intent_params = self.intent_classifier.classify_intent(rewritten_query)
        
        # Constraint extraction
        constraints = self.constraint_extractor.extract_constraints(rewritten_query)
        
        # Retrieval routing
        raw_retrieved = self.router.route_query(intent, intent_params)
        
        # Apply geometric verification to place recognition matches if found
        verified_matches = []
        for match in raw_retrieved:
            # Simulate features for candidate comparison
            q_feat = np.ones(128)
            cand_feat = np.ones(128)
            success, inliers, residual = self.reranker.verify_geometry(q_feat, cand_feat)
            if success or intent == "locate-object": # Allow object search past rerank for mock ease
                match["inlier_count"] = inliers
                match["residual"] = residual
                verified_matches.append(match)
                
        # Goal state determination
        goal_state = self.goal_detector.evaluate_goal_state(intent, verified_matches, searched_coverage=0.98)
        
        # Decision allocation
        decision_action, decision_params = self.decision_engine.evaluate_next_step(goal_state, verified_matches)
        
        # Execute Spatial CoT Reasoning Chain
        reasoning_out = self.reasoner.process_cot(rewritten_query, intent, verified_matches, constraints)
        
        latency = (time.time() - start_time) * 1000.0
        
        return {
            "query": user_query,
            "rewritten_query": rewritten_query,
            "intent": intent,
            "constraints": constraints,
            "decision": decision_action,
            "answer": reasoning_out["answer"],
            "chain_of_thought": reasoning_out["chain_of_thought"],
            "provenance": reasoning_out["provenance"],
            "latency_ms": latency
        }
