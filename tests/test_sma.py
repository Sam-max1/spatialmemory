import os
import time
import pytest
import numpy as np
from pathlib import Path
from sma.infra.config import Config
from sma.memory.working_memory import WorkingMemory, KeyframeRecord
from sma.memory.episodic_memory import EpisodicMemory
from sma.memory.long_term_memory import LongTermMemory
from sma.memory.memory_manager import MemoryManager
from sma.perception.chunker import Chunker
from sma.reasoning.intent_classifier import IntentClassifier
from sma.reasoning.tool_caller import ToolCaller

@pytest.fixture
def test_dirs(tmp_path):
    """
    Fixture creating temporary database paths for isolated unit testing.
    """
    db_ep = tmp_path / "test_episodic.db"
    db_sg = tmp_path / "test_scene_graph.db"
    db_vx = tmp_path / "test_voxels.db"
    return db_ep, db_sg, db_vx

def test_working_memory_eviction():
    wm = WorkingMemory(max_keyframes=3)
    
    # Add keyframes
    kf1 = KeyframeRecord("kf1", time.time(), b"", np.zeros(10), np.zeros(3), np.eye(3), "METRIC", {}, 0.5, 0.5)
    kf2 = KeyframeRecord("kf2", time.time(), b"", np.zeros(10), np.zeros(3), np.eye(3), "METRIC", {}, 0.5, 0.5)
    kf3 = KeyframeRecord("kf3", time.time(), b"", np.zeros(10), np.zeros(3), np.eye(3), "METRIC", {}, 0.5, 0.5)
    kf4 = KeyframeRecord("kf4", time.time(), b"", np.zeros(10), np.zeros(3), np.eye(3), "METRIC", {}, 0.5, 0.5)

    ev1 = wm.add_keyframe(kf1)
    ev2 = wm.add_keyframe(kf2)
    ev3 = wm.add_keyframe(kf3)
    
    assert ev1 is None
    assert ev2 is None
    assert ev3 is None
    assert len(wm.state.keyframe_ring) == 3

    # Adding 4th triggers FIFO eviction of 1st keyframe
    ev4 = wm.add_keyframe(kf4)
    assert ev4 is not None
    assert ev4.frame_id == "kf1"
    assert len(wm.state.keyframe_ring) == 3
    assert wm.state.keyframe_ring[0].frame_id == "kf2"

def test_episodic_memory_logging(test_dirs):
    db_ep, _, _ = test_dirs
    em = EpisodicMemory(db_ep)
    
    session_id = "test_sess"
    em.start_session(session_id, time.time())
    
    # Log robot movement pose
    em.log_pose(session_id, time.time(), (1.2, 3.4, 0.5, 1.5))
    
    # Log sight
    em.log_sighting("sight_01", session_id, time.time(), "laptop", (1.5, 3.6, 0.8), 0.95, b"", "kf1")
    
    trajectory = em.get_session_trajectory(session_id)
    sightings = em.get_session_sightings(session_id)
    
    assert len(trajectory) == 1
    assert trajectory[0]["x"] == 1.2
    assert len(sightings) == 1
    assert sightings[0]["label"] == "laptop"

def test_long_term_memory_scene_graph(test_dirs):
    _, db_sg, db_vx = test_dirs
    ltm = LongTermMemory(db_sg, db_vx)
    
    # Upsert object
    ltm.upsert_node("node_obj_01", "object", "desk chair", (4.5, 2.3, 0.4), 0.9, time.time())
    ltm.upsert_node("node_room_01", "room", "Conference Room", (4.0, 2.0, 0.0), 1.0, time.time())
    
    # Link room and object
    ltm.add_edge("node_obj_01", "node_room_01", "in")
    
    nodes = ltm.get_all_nodes()
    edges = ltm.get_all_edges()
    
    assert len(nodes) == 2
    assert len(edges) == 1
    assert edges[0] == ("node_obj_01", "node_room_01", "in")

def test_chunker_novelty_gate():
    chunker = Chunker(novelty_threshold=0.3)
    
    emb1 = np.array([1.0, 0.0, 0.0])
    emb2 = np.array([0.0, 1.0, 0.0])  # Orogonal vector = distance of 1.0 > 0.3
    
    is_kf1, _ = chunker.evaluate_keyframe(emb1, 10, 10)
    assert is_kf1 is True  # Always first is keyframe
    
    is_kf2, metrics2 = chunker.evaluate_keyframe(emb2, 10, 10)
    assert is_kf2 is True
    assert metrics2["novelty"] == 1.0

    # Very similar vector: novelty close to 0
    emb3 = np.array([0.0, 0.99, 0.0])
    is_kf3, metrics3 = chunker.evaluate_keyframe(emb3, 10, 10)
    assert is_kf3 is False
    assert metrics3["novelty"] < 0.1

def test_intent_classification():
    classifier = IntentClassifier()
    
    intent1, params1 = classifier.classify_intent("where is the fire extinguisher?")
    assert intent1 == "locate-object"
    assert params1["label"] == "fire extinguisher"
    
    intent2, params2 = classifier.classify_intent("go to the hallway corridor")
    assert intent2 == "navigate-to"
    assert params2["destination"] == "hallway corridor"

def test_a_star_path_planning():
    caller = ToolCaller()
    
    # 5x5 empty grid
    grid = np.zeros((5, 5))
    path, success = caller.plan_path(grid, (0, 0), (4, 4))
    
    assert success is True
    assert len(path) == 5  # Diagonal moves: (0,0) -> (1,1) -> (2,2) -> (3,3) -> (4,4)
    assert path[0] == (0, 0)
    assert path[-1] == (4, 4)
    
    # Put a blocking wall
    grid[1, 0:3] = 1
    # Wall blocks direct paths
    path, success = caller.plan_path(grid, (0, 0), (2, 2))
    assert success is True
