"""Module B: Shot boundary detection using PySceneDetect 0.7+.

Supports both single video and batch processing of multiple videos.
"""

from pathlib import Path
from typing import List, Optional

import cv2
from loguru import logger

from vsde.config import config, FRAMES_DIR
from vsde.data.models import Shot, VideoMetadata, VideoType


class ShotSegmenter:
    """Detect shot boundaries in a single video using PySceneDetect."""

    def __init__(
        self,
        metadata: VideoMetadata,
        threshold: Optional[float] = None,
    ) -> None:
        self.metadata = metadata
        self.threshold = threshold if threshold is not None else config.SHOT_DETECTION_THRESHOLD

    def segment(self) -> List[Shot]:
        """Detect shot boundaries. Returns list of Shot objects."""
        from scenedetect import ContentDetector, SceneManager, open_video

        video_path = Path(self.metadata.video_path)
        video_stream = open_video(str(video_path))
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=self.threshold))

        scene_manager.detect_scenes(video=video_stream)
        scenes = scene_manager.get_scene_list()
        video_stream.close()

        shots: List[Shot] = []
        prefix = f"{self.metadata.video_id}"

        for i, scene in enumerate(scenes):
            # scene is a tuple of (start, end) FrameTimecode in scenedetect 0.7+
            start_tc = scene[0]
            end_tc = scene[1]
            start_frame = start_tc.get_frames()
            end_frame = end_tc.get_frames()
            start_sec = start_tc.get_seconds()
            end_sec = end_tc.get_seconds()

            shots.append(
                Shot(
                    shot_id=f"{prefix}_shot_{i+1:04d}",
                    video_id=self.metadata.video_id,
                    anime=self.metadata.anime,
                    episode=self.metadata.episode,
                    video_type=self.metadata.video_type,
                    video_source_label=self.metadata.video_source_label,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    duration_sec=end_sec - start_sec,
                    frame_count=end_frame - start_frame,
                )
            )

        return shots


class BatchShotSegmenter:
    """Segment multiple videos into shots in batch."""

    def __init__(self, threshold: Optional[float] = None) -> None:
        self.threshold = threshold if threshold is not None else config.SHOT_DETECTION_THRESHOLD

    def segment_all(self, metadata_list: List[VideoMetadata]) -> List[Shot]:
        """Segment all videos. Returns flat list of Shot objects."""
        all_shots: List[Shot] = []
        for meta in metadata_list:
            logger.info(f"Segmenting: {meta.video_id}")
            segmenter = ShotSegmenter(meta, threshold=self.threshold)
            shots = segmenter.segment()
            all_shots.extend(shots)
            logger.info(f"  {meta.video_id}: {len(shots)} shots detected")

        logger.info(f"Batch segment complete: {len(all_shots)} total shots")
        return all_shots

    def segment_by_type(
        self,
        target_metadata: List[VideoMetadata],
        baseline_metadata: List[VideoMetadata],
    ) -> tuple[List[Shot], List[Shot]]:
        """Segment target and baseline video groups separately."""
        target_shots = self.segment_all(target_metadata)
        baseline_shots = self.segment_all(baseline_metadata)
        return target_shots, baseline_shots
