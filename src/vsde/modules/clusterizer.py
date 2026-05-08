"""Module F: Style clustering using HDBSCAN + UMAP.

HDBSCAN automatically discovers the number of style clusters.
UMAP reduces high-dimensional embeddings to a visualization-friendly space.
"""

import numpy as np
from loguru import logger

from vsde.config import config


class Clusterizer:
    """Cluster shots by visual style using HDBSCAN on UMAP-reduced embeddings."""

    def __init__(
        self,
        embeddings: np.ndarray | None = None,
        umap_dim: int | None = None,
        min_cluster_size: int | None = None,
        min_samples: int | None = None,
    ) -> None:
        self.embeddings = embeddings
        self.umap_dim = umap_dim or config.UMAP_N_COMPONENTS
        self.min_cluster_size = min_cluster_size or config.HDBSCAN_MIN_CLUSTER_SIZE
        self.min_samples = min_samples or config.HDBSCAN_MIN_SAMPLES
        self._reduced: np.ndarray | None = None
        self._cluster_labels: np.ndarray | None = None

    def set_embeddings(self, embeddings: np.ndarray) -> None:
        """Set embedding matrix (N x 1792)."""
        self.embeddings = embeddings

    def reduce(self) -> np.ndarray:
        """Reduce embeddings to UMAP_dim using UMAP."""
        if self.embeddings is None:
            raise ValueError("Embeddings not set.")
        logger.info(f"Reducing {len(self.embeddings)} embeddings with UMAP ({self.umap_dim} dims)...")

        # Placeholder — UMAP will be imported and used here
        # import umap
        # reducer = umap.UMAP(n_components=self.umap_dim, random_state=42)
        # self._reduced = reducer.fit_transform(self.embeddings)
        self._reduced = self.embeddings[:, : self.umap_dim]
        logger.info("UMAP reduction complete.")
        return self._reduced

    def cluster(self) -> np.ndarray:
        """Run HDBSCAN on reduced embeddings. Returns cluster labels (-1 = noise)."""
        if self._reduced is None:
            self.reduce()

        logger.info("Running HDBSCAN clustering...")
        # Placeholder — HDBSCAN will be imported and used here
        # import hdbscan
        # clusterer = hdbscan.HDBSCAN(
        #     min_cluster_size=self.min_cluster_size,
        #     min_samples=self.min_samples,
        #     metric="euclidean",
        # )
        # self._cluster_labels = clusterer.fit_predict(self._reduced)
        n = len(self._reduced) if self._reduced is not None else 0
        self._cluster_labels = np.zeros(n, dtype=int)
        logger.info(f"Clustering complete: {len(set(self._cluster_labels))} clusters found.")
        return self._cluster_labels

    def get_cluster_stats(self) -> dict:
        """Return statistics about discovered clusters."""
        if self._cluster_labels is None:
            self.cluster()

        labels = self._cluster_labels
        unique, counts = np.unique(labels, return_counts=True)
        stats = {
            "total_shots": len(labels),
            "n_clusters": len(unique[unique != -1]),
            "noise_count": int(counts[unique == -1][0]) if -1 in unique else 0,
            "clusters": {
                str(int(label)): int(count)
                for label, count in zip(unique, counts)
                if label != -1
            },
        }
        return stats
