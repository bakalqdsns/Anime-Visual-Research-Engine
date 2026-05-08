"""Module D: Visual embedding with DINOv2 and CLIP.

Supports both target and baseline shots, with separate caching.
"""

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torchvision.transforms as T
from loguru import logger
from PIL import Image

from vsde.config import config, EMBEDDING_CACHE_DIR
from vsde.data.models import Shot


class Embedder:
    """Generate dual embeddings for shots using DINOv2 and CLIP.

    Pipeline:
      1. Load image from shot.frame_path
      2. DINOv2 (ViT-L/14) → CLS token (1024-dim)
      3. OpenCLIP (ViT-L/14) → semantic embedding (768-dim)
      4. Concatenate → 1792-dim vector per shot

    DINOv2 is loaded via torch.hub from facebookresearch/dinov2.
    OpenCLIP is loaded via open_clip.create_model_and_transforms.
    """

    _IMAGE_MEAN = (0.485, 0.456, 0.406)
    _IMAGE_STD = (0.229, 0.224, 0.225)
    _IMAGE_SIZE = 224

    def __init__(
        self,
        device: str | None = None,
        batch_size: int = 8,
        dinov2_repo: str = "facebookresearch/dinov2",
        dinov2_model: str = "dinov2_vitl14",
        clip_model: str = "ViT-L/14",
        clip_pretrained: str = "openai",
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.dinov2_repo = dinov2_repo
        self.dinov2_model_name = dinov2_model
        self.clip_model_name = clip_model
        self.clip_pretrained = clip_pretrained

        self._dinov2_model: Optional[torch.nn.Module] = None
        self._clip_model: Optional[torch.nn.Module] = None
        self._clip_preprocess: Optional[callable] = None
        self._transform: Optional[callable] = None

    def encode(self, shots: List[Shot]) -> np.ndarray:
        """Encode shots to concatenated DINOv2 + CLIP embeddings (1792-dim).

        Returns: ndarray of shape (N, 1792).
        """
        if not shots:
            return np.empty((0, config.EMBEDDING_DIM_CONCAT), dtype=np.float32)

        valid_shots = [
            s for s in shots if s.frame_path and Path(s.frame_path).exists()
        ]
        skipped = len(shots) - len(valid_shots)
        if skipped:
            logger.warning(f"Skipping {skipped} shots with missing frame_path")

        logger.info(
            f"Encoding {len(valid_shots)} shots on {self.device} "
            f"(batch_size={self.batch_size})..."
        )

        if self._dinov2_model is None:
            self._load_models()

        embeddings = []
        for i in range(0, len(valid_shots), self.batch_size):
            batch = valid_shots[i : i + self.batch_size]
            batch_emb = self._encode_batch(batch)
            embeddings.append(batch_emb)

            processed = min(i + self.batch_size, len(valid_shots))
            logger.debug(f"  [{processed}/{len(valid_shots)}] batches encoded")

        result = np.vstack(embeddings)
        logger.info(f"Encoding complete: shape {result.shape}")
        return result

    def _encode_batch(self, batch: List[Shot]) -> np.ndarray:
        """Encode a batch of shots: DINOv2 CLS (1024) + CLIP embedding (768)."""
        images = []
        for shot in batch:
            img = Image.open(shot.frame_path).convert("RGB")
            images.append(self._transform(img))

        batch_tensor = torch.stack(images).to(self.device)

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.device == "cuda"):
            dinov2_emb = self._dinov2_model(batch_tensor)
            clip_emb = self._clip_model.encode_image(batch_tensor)
            clip_emb = clip_emb / clip_emb.norm(dim=-1, keepdim=True)

        dinov2_np = dinov2_emb.cpu().float().numpy()
        clip_np = clip_emb.cpu().float().numpy()

        return np.concatenate([dinov2_np, clip_np], axis=1)

    def _load_models(self) -> None:
        """Load DINOv2 and OpenCLIP models lazily on first encode call."""
        import open_clip

        logger.info("Loading DINOv2 model (ViT-L/14)...")
        self._dinov2_model = torch.hub.load(
            self.dinov2_repo,
            self.dinov2_model_name,
        )
        self._dinov2_model = self._dinov2_model.to(self.device)
        self._dinov2_model.eval()

        logger.info("Loading OpenCLIP model (ViT-L/14)...")
        self._clip_model, _, _ = open_clip.create_model_and_transforms(
            self.clip_model_name,
            pretrained=self.clip_pretrained,
        )
        self._clip_model = self._clip_model.to(self.device)
        self._clip_model.eval()

        self._transform = T.Compose([
            T.Resize(self._IMAGE_SIZE, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(self._IMAGE_SIZE),
            T.ToTensor(),
            T.Normalize(mean=self._IMAGE_MEAN, std=self._IMAGE_STD),
        ])

        dinov2_params = sum(p.numel() for p in self._dinov2_model.parameters()) / 1e6
        clip_params = sum(p.numel() for p in self._clip_model.parameters()) / 1e6
        logger.info(
            f"Models loaded — DINOv2: {dinov2_params:.1f}M params, "
            f"CLIP: {clip_params:.1f}M params"
        )


class BatchEmbedder:
    """Encode target and baseline shots separately with proper caching.

    Each shot group is cached independently, enabling breakpoint resume:
      - If cache exists and shot_ids match → load from cache (fast path)
      - If cache is stale (different shot list or count) → re-encode and overwrite
    """

    def __init__(self, device: str | None = None, batch_size: int = 8) -> None:
        self._embedder = Embedder(device=device, batch_size=batch_size)

    def encode_all(
        self,
        target_shots: List[Shot],
        baseline_shots: List[Shot],
        target_video_id: str = "target",
        baseline_video_id: str = "baseline",
        use_cache: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encode both target and baseline shots.

        Args:
            target_shots: List of target shots
            baseline_shots: List of baseline shots
            target_video_id: Group label for target (used in cache key)
            baseline_video_id: Group label for baseline (used in cache key)
            use_cache: If True, load from/save to .npy cache

        Returns:
            (target_embeddings, baseline_embeddings) as ndarrays of shape (N, 1792)
        """
        target_emb = self._encode_with_cache(
            target_shots, target_video_id, "target", use_cache
        )
        baseline_emb = self._encode_with_cache(
            baseline_shots, baseline_video_id, "baseline", use_cache
        )
        return target_emb, baseline_emb

    def _encode_with_cache(
        self,
        shots: List[Shot],
        group_id: str,
        video_type: str,
        use_cache: bool,
    ) -> np.ndarray:
        cache_path = EMBEDDING_CACHE_DIR / f"{group_id}.npy"
        meta_path = EMBEDDING_CACHE_DIR / f"{group_id}.meta.json"

        if use_cache and cache_path.exists() and meta_path.exists():
            try:
                cached = np.load(cache_path)
                with open(meta_path) as f:
                    meta = json.load(f)
                cached_ids = set(meta.get("shot_ids", []))
                current_ids = set(s.shot_id for s in shots)
                if cached_ids == current_ids and cached.shape[0] == len(shots):
                    logger.info(
                        f"Loaded {cached.shape[0]} cached embeddings from {cache_path}"
                    )
                    return cached
                logger.warning(
                    f"Cache mismatch for '{group_id}': "
                    f"{cached.shape[0]} cached vs {len(shots)} current shots — re-encoding"
                )
            except (json.JSONDecodeError, KeyError):
                logger.warning(f"Corrupted cache meta for '{group_id}' — re-encoding")

        embeddings = self._embedder.encode(shots)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, embeddings)
        with open(meta_path, "w") as f:
            json.dump(
                {
                    "group_id": group_id,
                    "video_type": video_type,
                    "shot_ids": [s.shot_id for s in shots],
                    "shape": list(embeddings.shape),
                },
                f,
                indent=2,
            )
        logger.info(f"Saved {embeddings.shape[0]} embeddings to {cache_path}")
        return embeddings

    def encode_single(
        self,
        shots: List[Shot],
        group_id: str = "single",
        use_cache: bool = True,
    ) -> np.ndarray:
        """Encode a single group of shots (convenience method).

        Returns only the target embeddings (no baseline).
        """
        emb, _ = self.encode_all(shots, [], target_video_id=group_id, baseline_video_id="", use_cache=use_cache)
        return emb
