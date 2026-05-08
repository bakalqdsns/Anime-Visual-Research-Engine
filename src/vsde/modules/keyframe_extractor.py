"""Module C: Keyframe extraction from shots.

Supports separate storage for target and baseline frames.
"""

from pathlib import Path
from typing import List, Optional

import cv2
from loguru import logger

from vsde.config import FRAMES_DIR
from vsde.data.models import Shot, VideoMetadata, VideoType


class KeyframeExtractor:
    """Extract representative keyframes from shots.

    Strategy: use the middle frame of each shot as the representative.
    This preserves visual continuity within a shot since a single shot
    represents a single camera/motion state.
    """

    def __init__(
        self,
        shots: List[Shot],
        video_metadata: Optional[VideoMetadata] = None,
    ) -> None:
        self.shots = shots
        self._video_metadata = video_metadata
        self._fps_cache: dict[str, float] = {}

    def extract(self) -> List[Shot]:
        """Extract and save keyframes, updating Shot.frame_path in place.

        Returns the same Shot list with frame_path populated.
        """
        for shot in self.shots:
            fps = self._resolve_fps(shot)
            mid_sec = (shot.start_sec + shot.end_sec) / 2
            frame_num = int(mid_sec * fps)

            # Determine subfolder by video_type
            subfolder = "target" if shot.video_type == VideoType.TARGET else "baseline"
            frame_dir = FRAMES_DIR / subfolder
            frame_dir.mkdir(parents=True, exist_ok=True)

            output_path = frame_dir / f"{shot.shot_id}.jpg"

            # Extract frame
            success = self._extract_frame(shot, frame_num, output_path)
            shot.frame_path = str(output_path) if success else None

            logger.debug(f"Extracted frame {frame_num} → {output_path} [{shot.shot_id}]")

        return self.shots

    def _extract_frame(
        self, shot: Shot, frame_num: int, output_path: Path
    ) -> bool:
        """Extract a single frame at frame_num and save to output_path."""
        video_path = self._resolve_video_path(shot)
        if video_path is None or not Path(video_path).exists():
            logger.warning(f"Video not found for shot {shot.shot_id}: {video_path}")
            return False

        cap = cv2.VideoCapture(video_path)
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(str(output_path), frame)
                return True
            return False
        finally:
            cap.release()

    def _resolve_video_path(self, shot: Shot) -> Optional[str]:
        """Resolve the video path for a given shot."""
        for ext in [".mp4", ".mkv", ".avi", ".mov", ".webm"]:
            subfolder = "target" if shot.video_type == VideoType.TARGET else "baseline"
            candidate = FRAMES_DIR.parent / "raw" / subfolder / f"{shot.video_id}{ext}"
            if candidate.exists():
                return str(candidate)
        return None

    def _resolve_fps(self, shot: Shot) -> float:
        """Resolve FPS for the video this shot belongs to.

        Checks (in order):
        1. Cached metadata for this video_id
        2. VideoMetadata passed at construction time
        3. Quick OpenCV probe on the video file
        """
        if shot.video_id in self._fps_cache:
            return self._fps_cache[shot.video_id]

        # If VideoMetadata was provided at construction time
        if self._video_metadata is not None and self._video_metadata.video_id == shot.video_id:
            fps = self._video_metadata.fps
            self._fps_cache[shot.video_id] = fps
            return fps

        # Quick probe via OpenCV
        video_path = self._resolve_video_path(shot)
        if video_path:
            cap = cv2.VideoCapture(video_path)
            try:
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    if fps > 0:
                        self._fps_cache[shot.video_id] = fps
                        return fps
            finally:
                cap.release()

        logger.warning(
            f"Could not resolve FPS for shot {shot.shot_id}, defaulting to 24.0"
        )
        return 24.0


class BatchKeyframeExtractor:
    """Extract keyframes from multiple shot lists (target + baseline).

    Maintains a shared FPS cache across all videos for efficiency.
    """

    def __init__(self, video_metadata_list: Optional[List[VideoMetadata]] = None) -> None:
        """Initialize with optional metadata for fast FPS resolution.

        Args:
            video_metadata_list: Pre-loaded VideoMetadata list. When provided,
                avoids per-shot OpenCV probes to get FPS.
        """
        self._metadata_by_id: dict[str, VideoMetadata] = {}
        if video_metadata_list:
            for meta in video_metadata_list:
                self._metadata_by_id[meta.video_id] = meta

    def extract_all(
        self,
        target_shots: List[Shot],
        baseline_shots: List[Shot],
    ) -> tuple[List[Shot], List[Shot]]:
        """Extract frames for both target and baseline shots."""
        logger.info(f"Extracting {len(target_shots)} target keyframes...")
        target = KeyframeExtractor(target_shots).extract()

        logger.info(f"Extracting {len(baseline_shots)} baseline keyframes...")
        baseline = KeyframeExtractor(baseline_shots).extract()
        return target, baseline

    def extract(
        self,
        shots: List[Shot],
        video_metadata_list: Optional[List[VideoMetadata]] = None,
    ) -> List[Shot]:
        """Extract keyframes for a single shot list with optional metadata.

        Args:
            shots: List of shots to extract frames from.
            video_metadata_list: Optional VideoMetadata to speed up FPS resolution.

        Returns:
            Same shot list with frame_path populated.
        """
        extractor = KeyframeExtractor(shots, video_metadata=video_metadata_list)
        return extractor.extract()
