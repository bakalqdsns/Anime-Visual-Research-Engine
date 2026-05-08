"""VSDE processing modules."""

from vsde.modules.video_loader import (
    VideoLoader,
    load_target,
    load_baseline,
    load_default_dirs,
)
from vsde.modules.shot_segmenter import ShotSegmenter, BatchShotSegmenter
from vsde.modules.keyframe_extractor import KeyframeExtractor, BatchKeyframeExtractor
from vsde.modules.embedder import Embedder, BatchEmbedder
from vsde.modules.differential_engine import DifferentialEngine
from vsde.modules.baseline_builder import BaselineBuilder, BaselineMetadata
from vsde.modules.clusterizer import Clusterizer
from vsde.modules.knowledge_base import KnowledgeBase

__all__ = [
    # Video loading
    "VideoLoader",
    "load_target",
    "load_baseline",
    "load_default_dirs",
    # Shot segmentation
    "ShotSegmenter",
    "BatchShotSegmenter",
    # Keyframe extraction
    "KeyframeExtractor",
    "BatchKeyframeExtractor",
    # Embedding
    "Embedder",
    "BatchEmbedder",
    # Core engine
    "DifferentialEngine",
    "BaselineBuilder",
    "BaselineMetadata",
    "Clusterizer",
    "KnowledgeBase",
]
