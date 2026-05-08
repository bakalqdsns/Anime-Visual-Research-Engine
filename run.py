#!/usr/bin/env python
"""VSDE CLI — Visual Style Differential Engine

Usage examples:

    # Build a baseline from a reference anime (run once)
    python run.py build-baseline --input data/raw/baseline/hibike_ep01.mp4 --name hibike_baseline

    # Run full analysis on target anime (uses cached baseline)
    python run.py analyze --target data/raw/target/bakemonogatari_ep01.mp4 --baseline-name hibike_baseline

    # Full pipeline: build baseline + analyze target in one shot
    python run.py full \
        --target data/raw/target/bakemonogatari_ep01.mp4 \
        --baseline data/raw/baseline/hibike_ep01.mp4 \
        --baseline-name hibike_baseline

    # Skip keyframe extraction and re-use cached embeddings
    python run.py analyze --target data/raw/target/bakemonogatari_ep01.mp4 \
        --baseline-name hibike_baseline --skip-keyframes
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from vsde.config import (
    config,
    TARGET_VIDEO_DIR,
    BASELINE_VIDEO_DIR,
    FRAMES_DIR,
    BASELINE_CACHE_DIR,
    PROMPT_OUTPUT_DIR,
)
from vsde.data.models import VideoType
from vsde.modules.video_loader import VideoLoader
from vsde.modules.embedder import BatchEmbedder
from vsde.modules.baseline_builder import BaselineBuilder
from vsde.modules.differential_engine import DifferentialEngine


# ── Helpers ──────────────────────────────────────────────────────────


def _banner(msg: str) -> None:
    width = 60
    print(f"\n{'─' * width}")
    print(f"  {msg}")
    print(f"{'─' * width}")


def _info(msg: str) -> None:
    print(f"  [INFO]  {msg}")


def _ok(msg: str) -> None:
    print(f"  [  OK  ]  {msg}")


def _warn(msg: str) -> None:
    print(f"  [ WARN ]  {msg}")


def _resolve_video(
    path_arg: str | Path | None,
    default_dir: Path,
    video_type: VideoType,
    label: str,
) -> tuple[VideoLoader, Path]:
    """Resolve a video source and return (loader, resolved_path)."""
    if path_arg:
        p = Path(path_arg).expanduser().resolve()
    else:
        # Try to find any video in the default directory
        videos = sorted(
            f for f in default_dir.iterdir()
            if f.suffix.lower() in {".mp4", ".mkv", ".avi", ".mov", ".webm"}
        )
        if not videos:
            raise FileNotFoundError(
                f"No video found in default dir '{default_dir}'. "
                "Please specify --target / --baseline explicitly."
            )
        p = videos[0]

    loader = VideoLoader(p, video_type=video_type, label=label)
    return loader, p


# ── Pipeline steps ───────────────────────────────────────────────────


def step_load(loader: VideoLoader) -> list:
    _info(f"Loading: {loader.source.path}")
    meta = loader.load_all()
    _info(f"Loaded {len(meta)} video(s)")
    for m in meta:
        _info(f"  {m.video_id}  {m.resolution[0]}x{m.resolution[1]}  {m.fps:.2f}fps  {m.duration_sec:.0f}s")
    return meta


def step_segment(metadata_list: list) -> list:
    from vsde.modules.shot_segmenter import BatchShotSegmenter
    seg = BatchShotSegmenter()
    shots = seg.segment_all(metadata_list)
    _ok(f"Segmented into {len(shots)} shots")
    return shots


def step_keyframes(shots: list) -> list:
    from vsde.modules.keyframe_extractor import KeyframeExtractor

    missing = [s for s in shots if not s.frame_path]
    if not missing:
        _info("Keyframes already cached — skipping extraction")
        return shots

    _info(f"Extracting {len(missing)} keyframes...")
    extractor = KeyframeExtractor(missing)
    extractor.extract()

    extracted = sum(1 for s in missing if s.frame_path)
    _ok(f"Extracted {extracted}/{len(missing)} keyframes")
    return shots


def step_embedding(
    target_shots: list,
    baseline_shots: list,
    target_video_id: str,
    baseline_video_id: str,
    use_cache: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    _info("Computing embeddings (DINOv2 + CLIP, cached)...")
    batch_emb = BatchEmbedder()
    target_emb, baseline_emb = batch_emb.encode_all(
        target_shots,
        baseline_shots,
        target_video_id=target_video_id,
        baseline_video_id=baseline_video_id,
        use_cache=use_cache,
    )
    _ok(f"Embeddings: target={target_emb.shape}, baseline={baseline_emb.shape}")
    return target_emb, baseline_emb


def step_build_baseline(
    baseline_embeddings: np.ndarray,
    baseline_shot_ids: list[str],
    baseline_name: str,
    strategy: str = "mean",
) -> tuple[np.ndarray, Path]:
    _info(f"Building baseline '{baseline_name}' (strategy={strategy}) from {len(baseline_shot_ids)} shots...")
    builder = BaselineBuilder(cache_dir=BASELINE_CACHE_DIR)
    builder.load_embeddings(baseline_embeddings, baseline_shot_ids, name=baseline_name)
    baseline_vec, path = builder.build_and_save(baseline_name, strategy=strategy)
    _ok(f"Baseline saved to {path}  (dim={baseline_vec.shape[0]})")
    return baseline_vec, path


def step_vs_baseline(
    target_embeddings: np.ndarray,
    target_shots: list,
    baseline_vec: np.ndarray,
    baseline_name: str,
    top_n: int = 200,
) -> list:
    _info(f"Running vs_baseline analysis (top {top_n} deviations)...")
    engine = DifferentialEngine(target_embeddings=target_embeddings)
    engine.set_baseline_vector(baseline_vec, name=baseline_name)
    engine.set_shots(target_shots=target_shots, baseline_shots=[])

    results = engine.compare(mode="vs_baseline")
    results = results[:top_n]

    _ok(f"Analysis complete — {len(results)} shots above threshold")
    if results:
        top = results[0]
        _info(f"  Most deviant: {top.shot_id}  magnitude={top.diff_magnitude:.4f}")

    return results


def step_cross_pairs(
    target_embeddings: np.ndarray,
    baseline_embeddings: np.ndarray,
    target_shots: list,
    baseline_shots: list,
    top_n: int = 100,
) -> list:
    _info(f"Running cross-group comparison (top {top_n} pairs)...")
    engine = DifferentialEngine(
        target_embeddings=target_embeddings,
        baseline_embeddings=baseline_embeddings,
    )
    engine.set_shots(target_shots=target_shots, baseline_shots=baseline_shots)
    pairs = engine.compare(mode="cross", n_pairs=top_n)
    _ok(f"Cross comparison: {len(pairs)} pairs")
    return pairs


def step_export_prompts(engine: DifferentialEngine, pairs: list, output_name: str) -> Path:
    _info(f"Exporting LLM prompts to {PROMPT_OUTPUT_DIR / output_name}/...")
    path = engine.export_prompts(pairs, output_name=output_name)
    _ok(f"Exported {len(pairs)} prompt files")
    return path


def step_summary(
    target_shots: list,
    baseline_shots: list,
    target_embeddings: np.ndarray,
    baseline_embeddings: np.ndarray,
    diff_results: list,
    cross_results: list | None,
    baseline_name: str,
) -> None:
    _banner("Results Summary")
    print(f"  Target shots       : {len(target_shots)}")
    print(f"  Baseline shots    : {len(baseline_shots)}")
    print(f"  Target embedding  : {target_embeddings.shape}")
    print(f"  Baseline embedding: {baseline_embeddings.shape}")
    print(f"  Baseline name     : {baseline_name}")
    print(f"  vs_baseline count : {len(diff_results)}")
    if cross_results:
        print(f"  Cross pairs count : {len(cross_results)}")

    if diff_results:
        print()
        print("  Top-5 most deviant shots:")
        for i, r in enumerate(diff_results[:5], 1):
            print(f"    {i}. {r.shot_id:<45} mag={r.diff_magnitude:.4f}")


# ── Commands ─────────────────────────────────────────────────────────


def cmd_build_baseline(args: argparse.Namespace) -> None:
    _banner("Build Baseline")
    loader, path = _resolve_video(args.input, BASELINE_VIDEO_DIR, VideoType.BASELINE, args.name)

    meta = step_load(loader)
    shots = step_segment(meta)
    shots = step_keyframes(shots)
    emb_batch = BatchEmbedder()
    emb = emb_batch.encode_single(shots, group_id=args.name, use_cache=True)
    step_build_baseline(emb, [s.shot_id for s in shots], args.name, strategy=args.strategy)


def cmd_analyze(args: argparse.Namespace) -> None:
    _banner("Analyze Target")
    start = time.perf_counter()

    # Load target
    target_loader, _ = _resolve_video(args.target, TARGET_VIDEO_DIR, VideoType.TARGET, args.target_label)
    target_meta = step_load(target_loader)
    target_shots = step_segment(target_meta)

    if not args.skip_keyframes:
        target_shots = step_keyframes(target_shots)

    # Load baseline
    if not args.baseline_name:
        raise ValueError("--baseline-name is required for analyze mode.")

    try:
        baseline_meta = BaselineBuilder.load(args.baseline_name, cache_dir=BASELINE_CACHE_DIR)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Baseline '{args.baseline_name}' not found in {BASELINE_CACHE_DIR}. "
            f"Run: python run.py build-baseline --input <baseline_video> --name {args.baseline_name}"
        )

    baseline_vec = baseline_meta.baseline_vector
    _info(f"Baseline loaded: '{baseline_meta.name}'  ({baseline_meta.shot_count} shots, strategy={baseline_meta.strategy})")

    # We still need baseline shots for cross comparison
    baseline_loader, _ = _resolve_video(
        args.baseline_video, BASELINE_VIDEO_DIR, VideoType.BASELINE, "baseline_for_cross"
    )
    baseline_meta_list = step_load(baseline_loader)
    baseline_shots = step_segment(baseline_meta_list)

    # Embeddings
    target_emb, baseline_emb = step_embedding(
        target_shots, baseline_shots,
        target_video_id=target_loader.label or target_meta[0].video_id,
        baseline_video_id=args.baseline_name,
        use_cache=not args.skip_keyframes,
    )

    # vs_baseline
    diff_results = step_vs_baseline(target_emb, target_shots, baseline_vec, args.baseline_name, top_n=args.top_n)

    # Cross pairs
    cross_results = step_cross_pairs(target_emb, baseline_emb, target_shots, baseline_shots, top_n=args.top_n // 2)

    # Export
    if args.export:
        engine = DifferentialEngine(target_embeddings=target_emb, baseline_embeddings=baseline_emb)
        engine.set_shots(target_shots, baseline_shots)
        step_export_prompts(engine, cross_results, output_name=args.export)

    step_summary(
        target_shots, baseline_shots,
        target_emb, baseline_emb,
        diff_results, cross_results,
        baseline_name=args.baseline_name,
    )

    elapsed = time.perf_counter() - start
    _ok(f"Done in {elapsed:.1f}s")


def cmd_full(args: argparse.Namespace) -> None:
    _banner("Full Pipeline: Build Baseline + Analyze Target")
    start = time.perf_counter()

    # ── Baseline ────────────────────────────────────────────────────
    _info("=== PHASE 1: Build baseline ===")
    bl_loader, _ = _resolve_video(args.baseline, BASELINE_VIDEO_DIR, VideoType.BASELINE, args.baseline_name)
    bl_meta = step_load(bl_loader)
    bl_shots = step_segment(bl_meta)
    bl_shots = step_keyframes(bl_shots)

    bl_emb_batch = BatchEmbedder()
    bl_emb, _ = bl_emb_batch.encode_all(bl_shots, [], target_video_id=args.baseline_name, baseline_video_id="", use_cache=True)
    baseline_vec, bl_path = step_build_baseline(bl_emb, [s.shot_id for s in bl_shots], args.baseline_name, strategy=args.strategy)

    # ── Target ──────────────────────────────────────────────────────
    _info("=== PHASE 2: Analyze target ===")
    tg_loader, _ = _resolve_video(args.target, TARGET_VIDEO_DIR, VideoType.TARGET, args.target_label or "target")
    tg_meta = step_load(tg_loader)
    tg_shots = step_segment(tg_meta)
    tg_shots = step_keyframes(tg_shots)

    tg_emb_batch = BatchEmbedder()
    tg_emb, _ = tg_emb_batch.encode_all(tg_shots, [], target_video_id=tg_loader.label or tg_meta[0].video_id, baseline_video_id="", use_cache=True)

    # vs_baseline
    diff_results = step_vs_baseline(tg_emb, tg_shots, baseline_vec, args.baseline_name, top_n=args.top_n)

    # Cross pairs (need baseline shots for cross)
    cross_results = step_cross_pairs(tg_emb, bl_emb, tg_shots, bl_shots, top_n=args.top_n // 2)

    # Export
    if args.export:
        engine = DifferentialEngine(target_embeddings=tg_emb, baseline_embeddings=bl_emb)
        engine.set_shots(tg_shots, bl_shots)
        step_export_prompts(engine, cross_results, output_name=args.export)

    step_summary(
        tg_shots, bl_shots,
        tg_emb, bl_emb,
        diff_results, cross_results,
        baseline_name=args.baseline_name,
    )

    elapsed = time.perf_counter() - start
    _ok(f"Full pipeline done in {elapsed:.1f}s")


# ── CLI argument parser ───────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python run.py",
        description="VSDE — Visual Style Differential Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── build-baseline ───────────────────────────────────────────────
    p_build = sub.add_parser(
        "build-baseline",
        help="Build a baseline center vector from a reference anime",
    )
    p_build.add_argument(
        "--input", "-i", type=str, default=None,
        help="Path to video file or folder. Defaults to first video in data/raw/baseline/",
    )
    p_build.add_argument(
        "--name", "-n", type=str, required=True,
        help="Baseline name (used for caching, e.g. 'hibike_baseline')",
    )
    p_build.add_argument(
        "--strategy", "-s", type=str, default="mean",
        choices=["mean", "trimmed_mean", "pca_center"],
        help="Aggregation strategy (default: mean)",
    )

    # ── analyze ──────────────────────────────────────────────────────
    p_analyze = sub.add_parser(
        "analyze",
        help="Analyze a target anime against an existing baseline",
    )
    p_analyze.add_argument(
        "--target", "-t", type=str, default=None,
        help="Path to target video file or folder. Defaults to first video in data/raw/target/",
    )
    p_analyze.add_argument(
        "--target-label", type=str, default="",
        help="Label for target group (used in cache key)",
    )
    p_analyze.add_argument(
        "--baseline-name", "-b", type=str, required=True,
        help="Name of the baseline to compare against",
    )
    p_analyze.add_argument(
        "--baseline-video", type=str, default=None,
        help="Baseline video path for cross comparison (optional, uses default dir if omitted)",
    )
    p_analyze.add_argument(
        "--top-n", type=int, default=200,
        help="Number of top-deviation shots to report (default: 200)",
    )
    p_analyze.add_argument(
        "--skip-keyframes", action="store_true",
        help="Skip keyframe extraction — re-use existing cache",
    )
    p_analyze.add_argument(
        "--export", type=str, default=None,
        help="Export LLM prompts to data/output/prompts/<name>/ (pass output name here)",
    )

    # ── full ─────────────────────────────────────────────────────────
    p_full = sub.add_parser(
        "full",
        help="Build baseline AND analyze target in one run",
    )
    p_full.add_argument(
        "--target", "-t", type=str, default=None,
        help="Path to target video file or folder",
    )
    p_full.add_argument(
        "--target-label", type=str, default="",
        help="Label for target group",
    )
    p_full.add_argument(
        "--baseline", "-b", type=str, default=None,
        help="Path to baseline reference video file or folder",
    )
    p_full.add_argument(
        "--baseline-name", "-n", type=str, required=True,
        help="Name for the baseline (used for caching)",
    )
    p_full.add_argument(
        "--strategy", "-s", type=str, default="mean",
        choices=["mean", "trimmed_mean", "pca_center"],
        help="Baseline aggregation strategy (default: mean)",
    )
    p_full.add_argument(
        "--top-n", type=int, default=200,
        help="Number of top-deviation shots to report (default: 200)",
    )
    p_full.add_argument(
        "--export", type=str, default=None,
        help="Export LLM prompts to data/output/prompts/<name>/",
    )

    return parser


# ── Entry point ───────────────────────────────────────────────────────


def main() -> None:
    config.log_settings()
    parser = _build_parser()
    args = parser.parse_args()

    try:
        if args.command == "build-baseline":
            cmd_build_baseline(args)
        elif args.command == "analyze":
            cmd_analyze(args)
        elif args.command == "full":
            cmd_full(args)
        else:
            parser.print_help()
            sys.exit(1)
    except FileNotFoundError as e:
        print(f"\n  [ERROR]  {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n  [ABORTED]  Interrupted by user")
        sys.exit(130)


if __name__ == "__main__":
    main()
