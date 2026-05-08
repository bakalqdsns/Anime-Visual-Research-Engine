"""Global configuration for VSDE."""

import os
from pathlib import Path

from loguru import logger

# Project root (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data"

# ── Video input paths ──────────────────────────────────────────────
# Target: the video(s) to be analyzed for visual style
TARGET_VIDEO_DIR = DATA_DIR / "raw" / "target"
# Baseline: reference video(s) used as comparison baseline
BASELINE_VIDEO_DIR = DATA_DIR / "raw" / "baseline"

# ── Output paths ───────────────────────────────────────────────────
# Extracted keyframes (both target and baseline)
FRAMES_DIR = DATA_DIR / "frames"
# Embedding cache
CACHE_DIR = DATA_DIR / "cache"
EMBEDDING_CACHE_DIR = CACHE_DIR / "embeddings"
# Baseline center vectors (from BaselineBuilder)
BASELINE_CACHE_DIR = CACHE_DIR / "baseline"
# Prompt output (LLM analysis prompts for shot pairs)
PROMPT_OUTPUT_DIR = DATA_DIR / "output" / "prompts"

# Ensure all runtime directories exist
for d in [
    TARGET_VIDEO_DIR,
    BASELINE_VIDEO_DIR,
    FRAMES_DIR / "target",
    FRAMES_DIR / "baseline",
    CACHE_DIR,
    EMBEDDING_CACHE_DIR,
    BASELINE_CACHE_DIR,
    PROMPT_OUTPUT_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)


class Config:
    """Global configuration object."""

    # Video processing
    SHOT_DETECTION_THRESHOLD = 27
    SHOT_DETECTION_METHOD = "detect-content"

    # Embedding models
    DINO_MODEL = "dinov2-vitl14"
    CLIP_MODEL = "ViT-L/14"
    EMBEDDING_DIM_DINO = 1024
    EMBEDDING_DIM_CLIP = 768
    EMBEDDING_DIM_CONCAT = EMBEDDING_DIM_DINO + EMBEDDING_DIM_CLIP

    # Clustering
    UMAP_N_COMPONENTS = 50
    HDBSCAN_MIN_CLUSTER_SIZE = 5
    HDBSCAN_MIN_SAMPLES = 3

    # Differential engine
    DIFF_TOP_N_PAIRS = 200
    DIFF_MIN_MAGNITUDE = 0.0

    # Database
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost:5432/vsde")
    EMBEDDING_VECTOR_SIZE = EMBEDDING_DIM_CONCAT

    # LLM
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")

    @classmethod
    def log_settings(cls) -> None:
        """Log current configuration."""
        logger.info("VSDE Configuration:")
        logger.info(f"  Project root : {PROJECT_ROOT}")
        logger.info(f"  Data dir     : {DATA_DIR}")
        logger.info(f"  Target videos: {TARGET_VIDEO_DIR}")
        logger.info(f"  Baseline vids: {BASELINE_VIDEO_DIR}")
        logger.info(f"  Frames dir   : {FRAMES_DIR}")
        logger.info(f"  Prompt output: {PROMPT_OUTPUT_DIR}")
        logger.info(f"  Cache dir    : {EMBEDDING_CACHE_DIR}")
        logger.info(f"  Shot thresh  : {cls.SHOT_DETECTION_THRESHOLD}")
        logger.info(f"  Embedding dim: {cls.EMBEDDING_DIM_CONCAT} "
                    f"(DINOv2 {cls.EMBEDDING_DIM_DINO} + CLIP {cls.EMBEDDING_DIM_CLIP})")


config = Config()
