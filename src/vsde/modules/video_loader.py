"""Module A: Video loading and metadata extraction.

Supports two input modes:
- Single file: one video path
- Folder: all video files in a directory
"""

from pathlib import Path
from typing import List

import cv2
from loguru import logger

from vsde.config import TARGET_VIDEO_DIR, BASELINE_VIDEO_DIR
from vsde.data.models import (
    Shot,
    VideoMetadata,
    VideoSource,
    VideoSourceType,
    VideoType,
)


# Supported video extensions
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".wmv", ".flv"}


def _is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def _scan_folder(folder: Path) -> List[Path]:
    """Return all video files in folder, non-recursive."""
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if _is_video(p))


class VideoLoader:
    """Load video file(s) and extract metadata.

    Accepts:
      - A single video file path
      - A folder path (scans all videos in it)
      - A VideoSource object
    """

    def __init__(
        self,
        source: str | Path | VideoSource,
        video_type: VideoType | None = None,
        label: str = "",
    ) -> None:
        self.source = self._resolve_source(source, video_type, label)
        self.video_type = self.source.video_type
        self.label = self.source.label

    def _resolve_source(
        self, source: str | Path | VideoSource,
        video_type: VideoType | None,
        label: str,
    ) -> VideoSource:
        if isinstance(source, VideoSource):
            return source
        p = Path(source).expanduser().resolve()
        if p.is_file():
            return VideoSource(
                path=str(p),
                source_type=VideoSourceType.SINGLE_FILE,
                video_type=video_type or VideoType.TARGET,
                label=label or p.stem,
            )
        elif p.is_dir():
            return VideoSource(
                path=str(p),
                source_type=VideoSourceType.FOLDER,
                video_type=video_type or VideoType.TARGET,
                label=label or p.name,
            )
        else:
            raise FileNotFoundError(f"Source not found: {p}")

    def load(self) -> VideoMetadata:
        """Load a single video file and return its metadata."""
        if self.source.source_type != VideoSourceType.SINGLE_FILE:
            raise ValueError(
                "load() only works on single-file sources. "
                "Use load_all() for folders."
            )
        return self._load_video(Path(self.source.path))

    def load_all(self) -> List[VideoMetadata]:
        """Load all videos from the source and return metadata list.

        If source is a single file, returns a list of one.
        If source is a folder, returns metadata for every video in it.
        """
        if self.source.source_type == VideoSourceType.SINGLE_FILE:
            return [self._load_video(Path(self.source.path))]

        folder = Path(self.source.path)
        video_paths = _scan_folder(folder)
        if not video_paths:
            logger.warning(f"No video files found in: {folder}")
            return []

        logger.info(f"Scanning folder: {folder} → {len(video_paths)} video(s) found")
        return [self._load_video(p) for p in video_paths]

    def _load_video(self, video_path: Path) -> VideoMetadata:
        """Extract metadata from a single video file."""
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        try:
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video: {video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration_sec = total_frames / fps if fps > 0 else 0.0

            # Parse episode from filename (e.g. bakemonogatari_ep01 → episode=1)
            episode = self._parse_episode(video_path.stem)

            return VideoMetadata(
                video_id=video_path.stem,
                video_path=str(video_path),
                anime=video_path.parent.name,
                episode=episode,
                fps=fps,
                duration_sec=duration_sec,
                total_frames=total_frames,
                resolution=(width, height),
                video_type=self.video_type,
                video_source_label=self.label,
            )
        finally:
            cap.release()

    def _parse_episode(self, video_stem: str) -> int:
        """Extract episode number from video filename.

        Supports patterns:
          bakemonogatari_ep01  → 1
          bakemonogatari_ep1   → 1
          bakemonogatari_01    → 1
          01_bakemonogatari    → 1
          bakemonogatari-s01e03 → 3
        """
        import re
        patterns = [
            r"ep(\d+)",        # ep01, ep1
            r"_(\d{2,})_",     # _01_, _12_
            r"s\d+e(\d+)",     # s01e03
            r"-(\d+)$",        # -01
        ]
        for pat in patterns:
            m = re.search(pat, video_stem, re.IGNORECASE)
            if m:
                return int(m.group(1))
        return 0


# ── Convenience constructors ────────────────────────────────────────


def load_target(
    source: str | Path | VideoSource,
    label: str = "",
) -> VideoLoader:
    """Create a VideoLoader for target video(s)."""
    return VideoLoader(source, video_type=VideoType.TARGET, label=label)


def load_baseline(
    source: str | Path | VideoSource,
    label: str = "",
) -> VideoLoader:
    """Create a VideoLoader for baseline video(s)."""
    return VideoLoader(source, video_type=VideoType.BASELINE, label=label)


def load_default_dirs(
    target_label: str = "",
    baseline_label: str = "",
) -> tuple[VideoLoader, VideoLoader]:
    """Create loaders pointing at the default TARGET and BASELINE directories."""
    target = VideoLoader(
        TARGET_VIDEO_DIR,
        video_type=VideoType.TARGET,
        label=target_label,
    )
    baseline = VideoLoader(
        BASELINE_VIDEO_DIR,
        video_type=VideoType.BASELINE,
        label=baseline_label,
    )
    return target, baseline


# ── Batch shot segmentation ─────────────────────────────────────────


def segment_videos(
    metadata_list: list[VideoMetadata],
    threshold: float | None = None,
) -> list[Shot]:
    """Segment a list of videos into shots using PySceneDetect.

    This is a convenience function that wraps BatchShotSegmenter.
    Use BatchShotSegmenter directly when you need more control.

    Args:
        metadata_list: List of VideoMetadata from VideoLoader.load_all()
        threshold: ContentDetector threshold (default: 27, per config)

    Returns:
        Flat list of Shot objects for all videos, ordered by video then shot index.
    """
    batch_seg = BatchShotSegmenter(threshold=threshold)
    return batch_seg.segment_all(metadata_list)


class BatchShotSegmenter:
    """Segment multiple videos into shots in batch.

    Processes all videos from a list of VideoMetadata objects.
    Supports threshold override per instance.
    """

    def __init__(self, threshold: float | None = None) -> None:
        from vsde.config import config
        self._threshold = threshold if threshold is not None else config.SHOT_DETECTION_THRESHOLD

    def segment_all(self, metadata_list: list[VideoMetadata]) -> list[Shot]:
        """Segment all videos in metadata_list. Returns flat list of Shot objects."""
        from vsde.modules.shot_segmenter import ShotSegmenter

        all_shots: list[Shot] = []
        for meta in metadata_list:
            logger.info(f"Segmenting: {meta.video_id}")
            segmenter = ShotSegmenter(meta, threshold=self._threshold)
            shots = segmenter.segment()
            all_shots.extend(shots)
            logger.info(f"  {meta.video_id}: {len(shots)} shots detected")

        logger.info(f"Batch segment complete: {len(all_shots)} total shots")
        return all_shots

    def segment_by_type(
        self,
        target_metadata: list[VideoMetadata],
        baseline_metadata: list[VideoMetadata],
    ) -> tuple[list[Shot], list[Shot]]:
        """Segment target and baseline video groups separately.

        Args:
            target_metadata: Metadata for target video(s)
            baseline_metadata: Metadata for baseline video(s)

        Returns:
            (target_shots, baseline_shots) as separate lists
        """
        target_shots = self.segment_all(target_metadata)
        baseline_shots = self.segment_all(baseline_metadata)
        return target_shots, baseline_shots
