import os
import time
import math
import random
import threading
import numpy as np
from flask import Flask, request, jsonify, render_template
from sma.infra.config import Config
from sma.reasoning.agent_runner import AgentRunner
from sma.infra.session_manager import SessionManager

app = Flask(__name__)
config = Config()
runner = AgentRunner(config)
session_mgr = SessionManager(runner)

# Global simulation state
sim_running = False
sim_thread = None
robot_pose = [5.0, 5.0, 0.0, 0.0]  # x, y, z, yaw

def seed_database():
    """
    Seed Long Term Memory with structured nodes representing rooms, surfaces, and objects
    to demonstrate scene graph indexing and query retrievals.
    """
    # Rooms
    runner.ltm.upsert_node("room_office_01", "room", "Office 101", (6.0, 6.0, 0.0), 1.0, time.time(), metadata={"floor": 1, "bounds": [0, 0, 12, 12]})
    runner.ltm.upsert_node("room_kitchen_01", "room", "Kitchen", (18.0, 15.0, 0.0), 1.0, time.time(), metadata={"floor": 1, "bounds": [12, 12, 24, 20]})
    runner.ltm.upsert_node("room_hallway_01", "room", "Corridor A", (10.0, 3.0, 0.0), 1.0, time.time(), metadata={"floor": 1, "bounds": [0, 0, 30, 5]})

    # Surfaces
    runner.ltm.upsert_node("surf_desk_01", "surface", "wooden desk", (5.0, 6.0, 0.7), 0.95, time.time())
    runner.ltm.upsert_node("surf_counter_01", "surface", "marble counter", (16.0, 14.5, 0.9), 0.95, time.time())

    # Objects
    runner.ltm.upsert_node("node_laptop_01", "object", "macbook laptop", (5.2, 5.9, 0.75), 0.90, time.time(), metadata={"source_session": "session_001"})
    runner.ltm.upsert_node("node_extinguisher_01", "object", "fire extinguisher", (2.0, 1.5, 1.2), 0.98, time.time(), metadata={"source_session": "session_001"})
    runner.ltm.upsert_node("node_chair_01", "object", "office chair", (4.8, 4.5, 0.4), 0.88, time.time(), metadata={"source_session": "session_002"})
    runner.ltm.upsert_node("node_fridge_01", "object", "refrigerator", (19.0, 16.0, 1.8), 0.99, time.time(), metadata={"source_session": "session_002"})
    runner.ltm.upsert_node("node_cup_01", "object", "coffee mug", (16.2, 14.3, 0.95), 0.85, time.time(), metadata={"source_session": "session_003"})

    # Edges
    # Objects ON Surfaces
    runner.ltm.add_edge("node_laptop_01", "surf_desk_01", "on")
    runner.ltm.add_edge("node_cup_01", "surf_counter_01", "on")
    
    # Objects / Surfaces IN Rooms
    runner.ltm.add_edge("surf_desk_01", "room_office_01", "in")
    runner.ltm.add_edge("node_chair_01", "room_office_01", "in")
    runner.ltm.add_edge("node_laptop_01", "room_office_01", "in")
    
    # Kitchen
    runner.ltm.add_edge("surf_counter_01", "room_kitchen_01", "in")
    runner.ltm.add_edge("node_cup_01", "room_kitchen_01", "in")
    runner.ltm.add_edge("node_fridge_01", "room_kitchen_01", "in")
    
    # Hallway
    runner.ltm.add_edge("node_extinguisher_01", "room_hallway_01", "in")

    # Connect Portals
    runner.ltm.upsert_node("portal_door_01", "portal", "Office Doorway", (12.0, 6.0, 0.0), 1.0, time.time())
    runner.ltm.add_edge("room_office_01", "portal_door_01", "connects_to")
    runner.ltm.add_edge("room_hallway_01", "portal_door_01", "connects_to")
    
    # Voxel seeds
    for offset in range(-5, 5):
        runner.ltm.set_voxel((100+offset, 120, 10), 0.2, 0.85, time.time())
        runner.ltm.set_voxel((320, 300+offset, 15), -0.1, 0.92, time.time())
        
    runner.ltm.save_voxels()

seed_database()

def run_simulation_loop():
    """
    Background simulation worker that generates dummy visual data and poses,
    driving the hot perception clock loop.
    """
    global sim_running, robot_pose
    t = 0.0
    while sim_running:
        t += 0.1
        # Circular exploration trajectory path
        # Map center ~ (15, 10)
        x = 15.0 + 8.0 * math.cos(t * 0.2)
        y = 10.0 + 5.0 * math.sin(t * 0.2)
        yaw = (t * 0.2) % (2 * math.pi)
        robot_pose = [x, y, 0.0, yaw]

        # Ingest mock camera frames
        dummy_img = f"FRAME_DATA_RAW_{int(time.time())}".encode('utf-8')
        runner.run_perception_step(dummy_img, (x, y, 0.0, yaw))
        
        time.sleep(0.1) # 10 FPS rate

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/status", methods=["GET"])
def get_status():
    nodes = runner.ltm.get_all_nodes()
    edges = runner.ltm.get_all_edges()
    trajectory = runner.em.get_session_trajectory(runner.current_session_id)
    
    return jsonify({
        "session_id": runner.current_session_id,
        "sim_running": sim_running,
        "robot_pose": robot_pose,
        "keyframes_count": len(runner.wm.state.keyframe_ring),
        "voxels_count": len(runner.ltm.voxel_cache),
        "scene_graph": {
            "nodes": [{
                "node_id": n["node_id"],
                "type": n["type"],
                "label": n["label"],
                "x": n["x"],
                "y": n["y"],
                "z": n["z"],
                "confidence": n["confidence"],
                "metadata": n["metadata"]
            } for n in nodes],
            "edges": [{"source": e[0], "target": e[1], "relation": e[2]} for e in edges]
        },
        "trajectory": trajectory[-100:] if trajectory else []
    })

@app.route("/api/simulation/start", methods=["POST"])
def start_simulation():
    global sim_running, sim_thread
    if not sim_running:
        sim_running = True
        sim_thread = threading.Thread(target=run_simulation_loop, daemon=True)
        sim_thread.start()
    return jsonify({"status": "running"})

@app.route("/api/simulation/stop", methods=["POST"])
def stop_simulation():
    global sim_running
    sim_running = False
    return jsonify({"status": "stopped"})

@app.route("/api/query", methods=["POST"])
def submit_query():
    data = request.json or {}
    query_text = data.get("query", "")
    if not query_text:
        return jsonify({"error": "Missing query field"}), 400

    # Execute intent re-routing and reasoning CoT
    response_payload = runner.run_query_step(query_text)
    return jsonify(response_payload)

@app.route("/api/plan", methods=["POST"])
def run_plan():
    data = request.json or {}
    start = tuple(data.get("start", [5, 5]))
    goal = tuple(data.get("goal", [40, 40]))
    
    # Convert list coordinates to tuples
    start_tuple = (int(start[0]), int(start[1]))
    goal_tuple = (int(goal[0]), int(goal[1]))
    
    path, success = runner.tool_caller.plan_path(runner.occupancy_grid, start_tuple, goal_tuple)
    return jsonify({
        "success": success,
        "path": path
    })

@app.route("/api/feedback", methods=["POST"])
def submit_feedback():
    data = request.json or {}
    node_id = data.get("node_id")
    corrected_label = data.get("label")
    
    if not node_id or not corrected_label:
        return jsonify({"error": "Missing node_id or label"}), 400
        
    # Execute label correction in LTM
    success = runner.ltm.upsert_node(
        node_id=node_id,
        node_type="object",
        label=corrected_label,
        position=(5.0, 6.0, 0.75), # Mock position
        confidence=1.0,
        last_seen=time.time(),
        metadata={"operator_corrected": True}
    )
    return jsonify({"success": True})

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=config.DEBUG)
