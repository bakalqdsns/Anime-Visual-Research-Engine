"""Module E: Differential Engine — core of VSDE.

Computes pairwise style difference vectors between shots.
Core insight: embedding_A - embedding_B = style_difference_vector

Supports:
- Within-group comparison (target vs target, baseline vs baseline)
- Cross-group comparison (target vs baseline)  ← primary mode for VSDE
"""

import heapq
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal, Optional, Union

import numpy as np
from loguru import logger

from vsde.config import config, PROMPT_OUTPUT_DIR
from vsde.data.models import (
    PromptOutput,
    Shot,
    ShotDiffAnalysis,
    ShotReference,
    ShotPairAnalysis,
    VideoType,
)


class DifferentialEngine:
    """Compute style difference vectors between shot pairs.

    Supports two comparison modes:
    - Pairwise: any two shots (shot vs shot), useful for within-group diversity
    - Baseline: each target shot vs a pre-computed baseline center vector
      (the primary mode for cross-work style analysis in VSDE)
    """

    def __init__(
        self,
        target_embeddings: Optional[np.ndarray] = None,
        baseline_embeddings: Optional[np.ndarray] = None,
    ) -> None:
        self.target_embeddings = target_embeddings
        self.baseline_embeddings = baseline_embeddings
        self._baseline_vector: Optional[np.ndarray] = None  # single center vector
        self._baseline_name: str = ""
        self._target_shots: List[Shot] = []
        self._baseline_shots: List[Shot] = []

    def set_baseline_vector(
        self,
        vector: np.ndarray,
        name: str = "",
    ) -> None:
        """Set a single baseline center vector (from BaselineBuilder).

        This enables the 'vs_baseline' comparison mode where every target shot
        is compared against the same center vector rather than individual shots.
        """
        if vector.ndim != 1:
            raise ValueError(
                f"baseline_vector must be 1D, got shape {vector.shape}"
            )
        self._baseline_vector = vector
        self._baseline_name = name
        logger.info(f"Baseline vector set (name='{name}', dim={vector.shape[0]})")

    def set_shots(
        self,
        target_shots: List[Shot],
        baseline_shots: List[Shot],
    ) -> None:
        """Set the shot lists for both groups."""
        self._target_shots = target_shots
        self._baseline_shots = baseline_shots

    def set_embeddings(
        self,
        target: np.ndarray,
        baseline: Optional[np.ndarray] = None,
    ) -> None:
        """Set embedding matrices. baseline is optional for within-group mode."""
        self.target_embeddings = target
        self.baseline_embeddings = baseline

    def compare(
        self,
        mode: Literal["cross", "within_target", "within_baseline", "all", "vs_baseline"] = "cross",
        n_pairs: Optional[int] = None,
        min_magnitude: Optional[float] = None,
    ) -> Union[List[ShotPairAnalysis], List[ShotDiffAnalysis]]:
        """Compare shot pairs and return analysis results.

        Modes:
            cross:          Target vs Baseline shots  ← primary for pairwise comparison
            vs_baseline:    Each target shot vs the baseline center vector  ← VSDE core
                             Returns List[ShotDiffAnalysis]
            within_target:  Target vs Target shots
            within_baseline: Baseline vs Baseline shots
            all:            All pairs (cross as main result)
        """
        n = n_pairs or config.DIFF_TOP_N_PAIRS
        min_mag = min_magnitude or config.DIFF_MIN_MAGNITUDE

        if mode == "vs_baseline":
            return self._compare_vs_baseline(min_mag)
        elif mode == "cross":
            pairs = self._find_cross_pairs(n, min_mag)
        elif mode == "within_target":
            pairs = self._find_within_pairs(self.target_embeddings, self._target_shots, n, min_mag)
        elif mode == "within_baseline":
            pairs = self._find_within_pairs(self.baseline_embeddings, self._baseline_shots, n, min_mag)
        else:  # all
            cross = self._find_cross_pairs(n, min_mag)
            return self._build_analysis(cross, n)

        return self._build_analysis(pairs, n)

    def _compare_vs_baseline(self, min_mag: float) -> List[ShotDiffAnalysis]:
        """Compare every target shot against the baseline center vector.

        Produces one ShotDiffAnalysis per target shot.
        This is the primary comparison mode for cross-work style analysis.
        """
        if self.target_embeddings is None:
            raise ValueError("target_embeddings must be set for vs_baseline mode.")
        if self._baseline_vector is None:
            raise ValueError(
                "baseline vector not set. Call set_baseline_vector() first."
            )

        if self.target_embeddings.shape[1] != len(self._baseline_vector):
            raise ValueError(
                f"Embedding dimension mismatch: target={self.target_embeddings.shape[1]}, "
                f"baseline={len(self._baseline_vector)}"
            )

        results: List[ShotDiffAnalysis] = []
        for i, shot in enumerate(self._target_shots):
            emb = self.target_embeddings[i]
            diff_vec = emb - self._baseline_vector
            diff_mag = float(np.linalg.norm(diff_vec))

            if diff_mag < min_mag:
                continue

            top_dims = [int(idx) for idx in np.abs(diff_vec).argsort()[-10:][::-1]]

            results.append(
                ShotDiffAnalysis(
                    shot_id=shot.shot_id,
                    baseline_name=self._baseline_name,
                    diff_vector=diff_vec.tolist(),
                    diff_magnitude=diff_mag,
                    cosine_similarity=float(
                        np.dot(emb, self._baseline_vector)
                        / (np.linalg.norm(emb) * np.linalg.norm(self._baseline_vector) + 1e-8)
                    ),
                    top_diff_dimensions=top_dims,
                )
            )

        results.sort(key=lambda r: r.diff_magnitude, reverse=True)
        logger.info(
            f"vs_baseline: {len(self._target_shots)} shots compared "
            f"against baseline '{self._baseline_name}' → "
            f"{len(results)} above threshold"
        )
        return results

    def _find_cross_pairs(
        self, n: int, min_mag: float
    ) -> List[tuple[int, int, float]]:
        """Find top-N most different target-baseline pairs.

        Compares every target shot against every baseline shot.
        Uses efficient sampling for large groups.
        """
        if self.target_embeddings is None or self.baseline_embeddings is None:
            raise ValueError("Both target and baseline embeddings must be set.")

        n_target = len(self.target_embeddings)
        n_baseline = len(self.baseline_embeddings)
        total_pairs = n_target * n_baseline

        logger.info(
            f"Cross comparison: {n_target} targets × {n_baseline} baselines "
            f"= {total_pairs:,} pairs → sampling top {n}"
        )

        if total_pairs <= n_target * n_baseline and total_pairs <= 10_000:
            return self._brute_cross_pairs(n_target, n_baseline, n, min_mag)
        return self._sample_cross_pairs(n_target, n_baseline, n, min_mag)

    def _brute_cross_pairs(
        self, n_target: int, n_baseline: int, n: int, min_mag: float
    ) -> List[tuple[int, int, float]]:
        """Brute-force all cross pairs (only for small datasets)."""
        heap: List[tuple[float, int, int]] = []
        for i in range(n_target):
            for j in range(n_baseline):
                mag = float(np.linalg.norm(
                    self.target_embeddings[i] - self.baseline_embeddings[j]
                ))
                if mag >= min_mag:
                    heapq.heappush(heap, (mag, i, j))
        return self._pop_heap(heap, n)

    def _sample_cross_pairs(
        self, n_target: int, n_baseline: int, n: int, min_mag: float
    ) -> List[tuple[int, int, float]]:
        """Randomly sample cross pairs and extract top-N by magnitude."""
        sample_size = min(n * 10, n_target * n_baseline)
        rng = np.random.default_rng()
        candidates: List[tuple[float, int, int]] = []

        for _ in range(sample_size):
            i = rng.integers(0, n_target)
            j = rng.integers(0, n_baseline)
            mag = float(np.linalg.norm(
                self.target_embeddings[i] - self.baseline_embeddings[j]
            ))
            if mag >= min_mag:
                heapq.heappush(candidates, (mag, i, j))

        return self._pop_heap(candidates, n)

    def _find_within_pairs(
        self,
        embeddings: np.ndarray,
        shots: List[Shot],
        n: int,
        min_mag: float,
    ) -> List[tuple[int, int, float]]:
        """Find top-N within-group pairs (shot vs shot in same group)."""
        n_shots = len(embeddings)
        logger.info(f"Within-group comparison: {n_shots} shots → sampling top {n}")

        if n_shots <= 100:
            return self._brute_within_pairs(embeddings, n, min_mag)
        return self._sample_within_pairs(embeddings, n, min_mag)

    def _brute_within_pairs(
        self, embeddings: np.ndarray, n: int, min_mag: float
    ) -> List[tuple[int, int, float]]:
        heap: List[tuple[float, int, int]] = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                mag = float(np.linalg.norm(embeddings[i] - embeddings[j]))
                if mag >= min_mag:
                    heapq.heappush(heap, (mag, i, j))
        return self._pop_heap(heap, n)

    def _sample_within_pairs(
        self, embeddings: np.ndarray, n: int, min_mag: float
    ) -> List[tuple[int, int, float]]:
        n_shots = len(embeddings)
        sample_size = min(n * 10, n_shots * (n_shots - 1) // 2)
        rng = np.random.default_rng()
        candidates: List[tuple[float, int, int]] = []
        drawn: set[tuple[int, int]] = set()

        while len(candidates) < sample_size:
            a = rng.integers(0, n_shots)
            b = rng.integers(0, n_shots)
            if a == b:
                continue
            pair = (min(a, b), max(a, b))
            if pair in drawn:
                continue
            drawn.add(pair)
            mag = float(np.linalg.norm(embeddings[a] - embeddings[b]))
            if mag >= min_mag:
                heapq.heappush(candidates, (mag, a, b))

        return self._pop_heap(candidates, n)

    def _pop_heap(
        self, heap: List[tuple[float, int, int]], n: int
    ) -> List[tuple[int, int, float]]:
        result: List[tuple[int, int, float]] = []
        for _ in range(n):
            if not heap:
                break
            mag, a, b = heapq.heappop(heap)
            result.append((a, b, mag))
        return result

    def _build_analysis(
        self,
        pairs: List[tuple[int, int, float]],
        n: int,
    ) -> List[ShotPairAnalysis]:
        """Build ShotPairAnalysis list from index pairs."""
        results: List[ShotPairAnalysis] = []
        for (a_idx, b_idx, mag) in pairs:
            shot_a = self._target_shots[a_idx] if a_idx < len(self._target_shots) else None
            shot_b = self._baseline_shots[b_idx] if b_idx < len(self._baseline_shots) else self._target_shots[b_idx] if b_idx < len(self._target_shots) else None

            if shot_a is None or shot_b is None:
                continue

            diff_vec = (
                self.target_embeddings[a_idx] - self.baseline_embeddings[b_idx]
                if self.baseline_embeddings is not None
                else self.target_embeddings[a_idx] - self.target_embeddings[b_idx]
            )

            results.append(
                ShotPairAnalysis(
                    shot_a=shot_a.shot_id,
                    shot_b=shot_b.shot_id,
                    shot_a_video_type=shot_a.video_type,
                    shot_b_video_type=shot_b.video_type,
                    diff_vector=diff_vec.tolist(),
                    diff_magnitude=mag,
                )
            )
        return results

    def export_prompts(
        self,
        pairs: List[ShotPairAnalysis],
        output_name: str = "shot_pairs",
        save_images: bool = False,
    ) -> Path:
        """Export shot pairs as structured LLM prompt JSON files.

        Creates one JSON per pair under PROMPT_OUTPUT_DIR/{output_name}/.
        """
        out_dir = PROMPT_OUTPUT_DIR / output_name
        out_dir.mkdir(parents=True, exist_ok=True)

        for i, pair in enumerate(pairs):
            shot_a = self._find_shot(pair.shot_a)
            shot_b = self._find_shot(pair.shot_b)

            dominant_dims = self._top_diff_dims(
                np.array(pair.diff_vector), top_k=10
            )

            prompt = self._build_prompt(shot_a, shot_b, pair, dominant_dims)

            output = PromptOutput(
                pair_id=f"{output_name}_pair_{i+1:04d}",
                shot_a=ShotReference(
                    shot_id=shot_a.shot_id,
                    video_id=shot_a.video_id,
                    video=shot_a.video_id,
                    video_type=shot_a.video_type,
                    frame_path=shot_a.frame_path or "",
                    start_sec=shot_a.start_sec,
                    end_sec=shot_a.end_sec,
                ),
                shot_b=ShotReference(
                    shot_id=shot_b.shot_id,
                    video_id=shot_b.video_id,
                    video=shot_b.video_id,
                    video_type=shot_b.video_type,
                    frame_path=shot_b.frame_path or "",
                    start_sec=shot_b.start_sec,
                    end_sec=shot_b.end_sec,
                ),
                diff_vector_magnitude=pair.diff_magnitude,
                dominant_diff_direction=dominant_dims,
                prompt=prompt,
                created_at=datetime.now(timezone.utc).isoformat(),
            )

            out_path = out_dir / f"pair_{i+1:04d}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(output.model_dump_json(indent=2, ensure_ascii=False))

        logger.info(f"Saved {len(pairs)} prompt files to {out_dir}")
        return out_dir

    def _find_shot(self, shot_id: str) -> Shot:
        for s in list(self._target_shots) + list(self._baseline_shots):
            if s.shot_id == shot_id:
                return s
        raise ValueError(f"Shot not found: {shot_id}")

    def _top_diff_dims(self, diff_vec: np.ndarray, top_k: int = 10) -> List[int]:
        return [int(i) for i in np.abs(diff_vec).argsort()[-top_k:][::-1]]

    def _build_prompt(
        self,
        shot_a: Shot,
        shot_b: Shot,
        pair: ShotPairAnalysis,
        dominant_dims: List[int],
    ) -> dict:
        return {
            "system": (
                "You are an expert anime visual style analyst. "
                "Compare two anime shots and describe the precise visual differences "
                "in composition, lighting, color palette, framing, and emotional tone."
            ),
            "user": (
                f"## Shot A (target)\n"
                f"- Video: {shot_a.video_id}\n"
                f"- Time: {shot_a.start_sec:.1f}s - {shot_a.end_sec:.1f}s\n"
                f"- Frame: {shot_a.frame_path}\n\n"
                f"## Shot B (baseline)\n"
                f"- Video: {shot_b.video_id}\n"
                f"- Time: {shot_b.start_sec:.1f}s - {shot_b.end_sec:.1f}s\n"
                f"- Frame: {shot_b.frame_path}\n\n"
                f"## Quantitative Context\n"
                f"- Difference magnitude (L2): {pair.diff_magnitude:.4f}\n"
                f"- Highest-variance embedding dims: {dominant_dims}\n\n"
                f"## Focus Areas\n"
                f"Analyze and describe differences in:\n"
                f"1. **Composition** — framing, rule of thirds, centering vs offset\n"
                f"2. **Lighting** — brightness, contrast, light source direction\n"
                f"3. **Color palette** — dominant hues, saturation, color harmony\n"
                f"4. **Framing** — shot scale (close-up/wide), camera angle implication\n"
                f"5. **Emotional tone** — calm vs dynamic, text density, visual pacing\n"
            ),
            "response_format": {
                "composition_diff": "string",
                "lighting_diff": "string",
                "color_diff": "string",
                "emotion_diff": "string",
                "overall_style_shift": "string",
            },
        }
