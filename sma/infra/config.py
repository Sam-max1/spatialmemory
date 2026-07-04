import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()

class Config:
    WORKSPACE_DIR = Path("/source/python/sammax1/embodiedai")
    DATA_DIR = WORKSPACE_DIR / "data"
    LOGS_DIR = WORKSPACE_DIR / "logs"

    # Setup directories
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    SITE_ID = os.getenv("SITE_ID", "site_office_01")
    MODEL_PATHS = os.getenv("MODEL_PATHS", "models/vpr_mega_loc.onnx:models/owl_v2.onnx").split(":")
    SENSOR_CONFIG = os.getenv("SENSOR_CONFIG", "configs/camera_monocular.json")

    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    OFFLINE_MODE = os.getenv("OFFLINE_MODE", "true").lower() in ("true", "1", "yes")
    STORAGE_BUDGET_GB = float(os.getenv("STORAGE_BUDGET_GB", "20"))
    PRIVACY_MODE = os.getenv("PRIVACY_MODE", "true").lower() in ("true", "1", "yes")
    DEBUG = os.getenv("DEBUG", "true").lower() in ("true", "1", "yes")

    # DB Paths
    SQLITE_EPISODIC_PATH = DATA_DIR / f"{SITE_ID}_episodic.db"
    SQLITE_SCENE_GRAPH_PATH = DATA_DIR / f"{SITE_ID}_scene_graph.db"
    ROCKSDB_VOXEL_PATH = DATA_DIR / f"{SITE_ID}_voxels.db"  # Directory path for voxel DB
    FAISS_PLACE_INDEX_PATH = DATA_DIR / f"{SITE_ID}_faiss_places.bin"
    FAISS_OBJECT_INDEX_PATH = DATA_DIR / f"{SITE_ID}_faiss_objects.bin"
