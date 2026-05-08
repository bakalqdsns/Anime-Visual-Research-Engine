"""Module B: Shot boundary detection using PySceneDetect.

Supports both single video and batch processing of multiple videos.
"""

from pathlib import Path
from typing import List, Optional

import cv2
from loguru import logger
from scenedetect import SceneManager, VideoManager
from scenedetect.detectors.content_detector import ContentDetector

from vsde.config import config, FRAMES_DIR
from vsde.data.models import Shot, VideoMetadata, VideoType


class ShotSegmenter:
    """Detect shot boundaries in video(s) using PySceneDetect."""

    def __init__(
        self,
        metadata: VideoMetadata,
        threshold: Optional[float] = None,
    ) -> None:
        self.metadata = metadata
        self.threshold = threshold or config.SHOT_DETECTION_THRESHOLD

    def segment(self) -> List[Shot]:
        """Detect shot boundaries for a single video. Returns list of Shot objects."""
        video_path = Path(self.metadata.video_path)
        video_manager = VideoManager([str(video_path)])
        scene_manager = SceneManager()
        scene_manager.add_detector(
            ContentDetector(threshold=self.threshold)
        )

        video_manager.set_duration()
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        scenes = scene_manager.get_scene_list()
        video_manager.release()

        shots: List[Shot] = []
        prefix = f"{self.metadata.video_id}"

        for i, scene in enumerate(scenes):
            start_frame = scene[0].get_frames()
            end_frame = scene[1].get_frames()
            start_sec = start_frame / self.metadata.fps
            end_sec = end_frame / self.metadata.fps

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
    """Segment multiple videos into shots in batch.

    Processes all videos from a list of VideoMetadata objects.
    """

    def __init__(
        self,
        threshold: Optional[float] = None,
    ) -> None:
        self.threshold = threshold or config.SHOT_DETECTION_THRESHOLD

    def segment_all(
        self, metadata_list: List[VideoMetadata]
    ) -> List[Shot]:
        """Segment all videos in metadata_list. Returns flat list of Shot objects."""
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
        """Segment target and baseline videos separately."""
        target_shots = self.segment_all(target_metadata)
        baseline_shots = self.segment_all(baseline_metadata)
        return target_shots, baseline_shots
