"""Pydantic data models for VSDE entities."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class VideoType(str, Enum):
    """Video role in the differential analysis pipeline."""

    TARGET = "target"
    BASELINE = "baseline"


class VideoSourceType(str, Enum):
    """Whether a video source is a single file or a folder of files."""

    SINGLE_FILE = "single_file"
    FOLDER = "folder"


class VideoSource(BaseModel):
    """A source of video files — either a single file or a folder.

    Both target and baseline can contain multiple videos.
    """

    path: str = Field(description="Absolute path to file or folder")
    source_type: VideoSourceType
    video_type: VideoType = VideoType.TARGET
    label: str = Field(
        default="",
        description="Human-readable label, e.g. 'bakemonogatari_s1' or 'standard_animation'",
    )

    model_config = {"frozen": True}


class VideoMetadata(BaseModel):
    """Metadata extracted from a single video file."""

    video_id: str = Field(description="Unique ID, e.g. bakemonogatari_ep01")
    video_path: str = Field(description="Absolute path to the video file")
    anime: str = Field(description="Anime series name")
    episode: int = Field(default=0)
    fps: float
    duration_sec: float
    total_frames: int
    resolution: tuple[int, int]
    video_type: VideoType = VideoType.TARGET
    video_source_label: str = Field(
        default="",
        description="Label of the VideoSource this video belongs to",
    )

    model_config = {"frozen": True}


class Shot(BaseModel):
    """A single shot (continuous sequence of frames between cuts)."""

    shot_id: str = Field(
        description="Unique identifier, e.g. bakemonogatari_ep01_shot_0001"
    )
    video_id: str = Field(description="Parent video ID")
    anime: str
    episode: int
    video_type: VideoType = VideoType.TARGET
    video_source_label: str = Field(default="")
    start_sec: float = Field(description="Start time in seconds")
    end_sec: float = Field(description="End time in seconds")
    duration_sec: float = Field(description="Duration in seconds")
    frame_count: int = Field(description="Number of frames in this shot")
    frame_path: Optional[str] = Field(
        default=None,
        description="Path to extracted representative frame",
    )


class ShotReference(BaseModel):
    """Lightweight reference to a shot, used in pair analysis."""

    shot_id: str
    video_id: str
    video: str = Field(description="Video filename")
    video_type: VideoType
    frame_path: str
    start_sec: float
    end_sec: float


class PromptOutput(BaseModel):
    """LLM prompt output for a shot pair."""

    pair_id: str = Field(description="Unique pair identifier")
    shot_a: ShotReference
    shot_b: ShotReference
    diff_vector_magnitude: float = Field(description="L2 norm of the diff vector")
    dominant_diff_direction: Optional[list[int]] = Field(
        default=None,
        description="Indices of highest-magnitude dimensions in the diff vector",
    )
    focus_aspects: list[str] = Field(
        default=["composition", "lighting", "color_palette", "framing", "emotion"],
    )
    prompt: dict = Field(description="Structured prompt to send to LLM")
    created_at: str


class ShotPairAnalysis(BaseModel):
    """Analysis result for a pair of shots compared by the differential engine."""

    shot_a: str = Field(description="Shot ID of the first shot")
    shot_b: str = Field(description="Shot ID of the second shot (or baseline name for vs_baseline mode)")
    shot_a_video_type: VideoType
    shot_b_video_type: VideoType
    diff_vector: list[float] = Field(
        description="1792-dimensional difference vector (A - B or target - baseline_center)"
    )
    diff_magnitude: float = Field(description="L2 norm of the difference vector")

    composition_diff: list[str] = Field(default_factory=list)
    lighting_diff: list[str] = Field(default_factory=list)
    color_diff: list[str] = Field(default_factory=list)
    emotion_diff: list[str] = Field(default_factory=list)

    llm_analysis: Optional[dict] = Field(
        default=None,
        description="Structured analysis from a multi-modal LLM",
    )


class ShotDiffAnalysis(BaseModel):
    """Result of comparing a single target shot against the baseline center vector.

    Produced by DifferentialEngine in 'vs_baseline' mode.
    """

    shot_id: str = Field(description="Target shot ID")
    baseline_name: str = Field(
        default="",
        description="Name of the baseline reference (e.g. 'hibike_baseline')",
    )
    diff_vector: list[float] = Field(
        description="Difference vector: embedding_target - baseline_center"
    )
    diff_magnitude: float = Field(
        description="L2 norm — absolute style deviation from baseline"
    )
    cosine_similarity: float = Field(
        default=0.0,
        description="Cosine similarity between shot embedding and baseline",
    )
    top_diff_dimensions: list[int] = Field(
        default_factory=list,
        description="Indices of embedding dimensions with largest absolute deviation",
    )
    llm_analysis: Optional[dict] = Field(
        default=None,
        description="Structured LLM analysis of the deviation",
    )
