"""Module H: Baseline Builder — computes statistical center vectors from reference works.

The baseline serves as the "neutral animation" reference point.
Each target shot is then compared against this center, producing a diff_vector
that reveals how much and in which direction the target deviates from the baseline.

Supported aggregation strategies:
- mean:        Simple arithmetic mean (recommended for MVP)
- trimmed_mean: Trimmed mean (robust to extreme shots in the reference work)
- pca_center:  PCA mean (robust in high-dimensional space)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from loguru import logger

from vsde.config import config, EMBEDDING_CACHE_DIR


@dataclass
class BaselineMetadata:
    """Metadata describing a computed baseline reference."""

    name: str
    strategy: str
    embedding_dim: int
    shot_count: int
    baseline_vector: np.ndarray = field(repr=False)
    created_at: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "strategy": self.strategy,
            "embedding_dim": self.embedding_dim,
            "shot_count": self.shot_count,
            "baseline_vector": self.baseline_vector.tolist(),
            "created_at": self.created_at,
        }


class BaselineBuilder:
    """Build a statistical baseline center vector from a collection of shot embeddings.

    Usage:
        builder = BaselineBuilder()
        builder.load_embeddings_from_cache(video_id="hibike_ep01")
        baseline = builder.build(strategy="mean")
        builder.save("hibike_baseline", baseline)
    """

    def __init__(self, cache_dir: Path | str = EMBEDDING_CACHE_DIR) -> None:
        self.cache_dir = Path(cache_dir)
        self._embeddings: Optional[np.ndarray] = None
        self._shot_ids: list[str] = []

    # ── Embedding loading ──────────────────────────────────────────────────────

    def load_embeddings(
        self,
        embeddings: np.ndarray,
        shot_ids: list[str],
        name: str = "unnamed",
    ) -> None:
        """Load pre-computed embeddings directly (e.g. from embedder output)."""
        if embeddings.ndim != 2:
            raise ValueError(
                f"embeddings must be 2D (n_shots, dim), got shape {embeddings.shape}"
            )
        self._embeddings = embeddings
        self._shot_ids = shot_ids
        logger.info(
            f"Loaded {len(shot_ids)} embeddings with shape {embeddings.shape} for baseline '{name}'"
        )

    def load_embeddings_from_cache(
        self,
        video_id: str,
        model: str = "concat",
    ) -> bool:
        """Load cached embeddings for a given video_id.

        Looks for files matching:
            {cache_dir}/{video_id}_*.{model}.npy
        """
        pattern = f"{video_id}_*.{model}.npy"
        files = sorted(self.cache_dir.glob(pattern))

        if not files:
            logger.warning(f"No cached embeddings found for '{video_id}' with pattern '{pattern}'")
            return False

        emb_list: list[np.ndarray] = []
        loaded_ids: list[str] = []

        for f in files:
            emb = np.load(f)
            emb_list.append(emb)
            shot_id = f.stem.replace(f".{model}", "")
            loaded_ids.append(shot_id)

        self._embeddings = np.stack(emb_list) if emb_list else np.array([])
        self._shot_ids = loaded_ids
        logger.info(
            f"Loaded {len(self._shot_ids)} cached embeddings for baseline '{video_id}'"
        )
        return True

    # ── Baseline computation ────────────────────────────────────────────────────

    def build(
        self,
        strategy: Literal["mean", "trimmed_mean", "pca_center"] = "mean",
        trim_proportion: float = 0.1,
    ) -> np.ndarray:
        """Compute the baseline center vector.

        Args:
            strategy: Aggregation strategy.
                - "mean":          Simple arithmetic mean. Most stable.
                - "trimmed_mean":  Trimmed mean, drops top/bottom `trim_proportion` per dimension.
                - "pca_center":    Mean after PCA projection (experimental).
            trim_proportion: Fraction of extreme values to trim (only for "trimmed_mean").

        Returns:
            1D numpy array of shape (embedding_dim,).
        """
        if self._embeddings is None or len(self._embeddings) == 0:
            raise ValueError("No embeddings loaded. Call load_embeddings() first.")

        if strategy == "mean":
            baseline = self._compute_mean()
        elif strategy == "trimmed_mean":
            baseline = self._compute_trimmed_mean(trim_proportion)
        elif strategy == "pca_center":
            baseline = self._compute_pca_center()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        logger.info(
            f"Built baseline with strategy '{strategy}' "
            f"from {len(self._embeddings)} shots → shape {baseline.shape}"
        )
        return baseline

    def _compute_mean(self) -> np.ndarray:
        return np.mean(self._embeddings, axis=0)

    def _compute_trimmed_mean(self, proportion: float) -> np.ndarray:
        from scipy.stats import trim_mean

        result = np.array(
            [trim_mean(col, proportion) for col in self._embeddings.T]
        )
        return result

    def _compute_pca_center(self) -> np.ndarray:
        from sklearn.decomposition import PCA

        n_components = min(50, len(self._embeddings) - 1, self._embeddings.shape[1])
        pca = PCA(n_components=n_components)
        pca.fit(self._embeddings)
        return pca.mean_

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(
        self,
        name: str,
        baseline: np.ndarray,
        strategy: str = "mean",
        metadata: Optional[dict] = None,
    ) -> Path:
        """Save baseline vector and metadata to cache directory.

        Outputs:
            {cache_dir}/baseline_{name}.npy         — the vector
            {cache_dir}/baseline_{name}.meta.json    — metadata
        """
        out_dir = self.cache_dir / "baseline"
        out_dir.mkdir(parents=True, exist_ok=True)

        vec_path = out_dir / f"baseline_{name}.npy"
        meta_path = out_dir / f"baseline_{name}.meta.json"

        np.save(vec_path, baseline)
        logger.info(f"Saved baseline vector to {vec_path}")

        import json
        from datetime import datetime, timezone

        meta = {
            "name": name,
            "strategy": strategy,
            "embedding_dim": int(baseline.shape[0]),
            "shot_count": len(self._shot_ids),
            "shot_ids": self._shot_ids,
            **(metadata or {}),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved baseline metadata to {meta_path}")

        return vec_path

    @classmethod
    def load(cls, name: str, cache_dir: Path | str = EMBEDDING_CACHE_DIR) -> BaselineMetadata:
        """Load a saved baseline from cache."""
        cache_dir = Path(cache_dir) / "baseline"
        vec_path = cache_dir / f"baseline_{name}.npy"
        meta_path = cache_dir / f"baseline_{name}.meta.json"

        if not vec_path.exists():
            raise FileNotFoundError(f"Baseline not found: {vec_path}")

        import json

        baseline_vector = np.load(vec_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        return BaselineMetadata(
            name=meta["name"],
            strategy=meta["strategy"],
            embedding_dim=meta["embedding_dim"],
            shot_count=meta["shot_count"],
            baseline_vector=baseline_vector,
            created_at=meta.get("created_at", ""),
        )

    # ── Convenience: build-and-save in one call ───────────────────────────────

    def build_and_save(
        self,
        name: str,
        strategy: Literal["mean", "trimmed_mean", "pca_center"] = "mean",
    ) -> tuple[np.ndarray, Path]:
        """Compute and persist a baseline in a single call."""
        baseline = self.build(strategy=strategy)
        path = self.save(name, baseline, strategy=strategy)
        return baseline, path
