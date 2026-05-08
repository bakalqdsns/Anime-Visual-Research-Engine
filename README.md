# VSDE - Visual Style Differential Engine

**VSDE 的核心哲学不是"分类"，而是"发现差异"。**

传统 AI 分析问的是："这个镜头是什么风格？"
VSDE 问的是："这个镜头和基线相比，哪里不一样？"

## Architecture

```
Target Video ─┐
              ├─→ Shot Segmentation ─┐
Baseline Video─┘                      ├─→ Embedding (DINOv2 + CLIP)
                                   ↓
                         Differential Engine (A - B = diff_vector)
                                   ↓
                         Clustering → Style Knowledge Base
```

## Dual-Video Architecture

VSDE 需要**两组视频输入**，每组都可以是**单个文件或文件夹**：

| 类型 | 支持模式 | 路径 |
|------|----------|------|
| Target | 单文件 `video.mp4` 或 文件夹 `target/` | `TARGET_VIDEO_DIR` |
| Baseline | 单文件 `standard.mp4` 或 文件夹 `baseline/` | `BASELINE_VIDEO_DIR` |

**文件夹模式**：将多集/多部作品放入同一文件夹，系统自动扫描全部视频，统一处理后合并所有镜头。

差异分析始终在 **Target 镜头 vs Baseline 镜头** 之间进行（跨组比对），LLM Prompt 输出到 `PROMPT_OUTPUT_DIR`。

## Quick Start

```python
from vsde.modules import (
    VideoLoader, load_target, load_baseline, load_default_dirs,
    BatchShotSegmenter, BatchKeyframeExtractor, BatchEmbedder,
    DifferentialEngine,
)

# 方式 A：使用默认目录（target/ 和 baseline/ 文件夹）
target_loader, baseline_loader = load_default_dirs(
    target_label="bakemonogatari_s1",
    baseline_label="standard_animation",
)

# 方式 B：手动指定路径（单文件或文件夹均可）
# target_loader = load_target("data/raw/target/bakemonogatari_ep01.mp4")
# baseline_loader = load_baseline("data/raw/baseline/")

# ── 步骤 1：扫描视频 ──────────────────────────────────────────────
target_meta = target_loader.load_all()
baseline_meta = baseline_loader.load_all()

# ── 步骤 2：镜头切分（批量） ──────────────────────────────────────
batch_seg = BatchShotSegmenter()
target_shots, baseline_shots = batch_seg.segment_by_type(target_meta, baseline_meta)

# ── 步骤 3：关键帧提取（批量） ───────────────────────────────────
batch_kf = BatchKeyframeExtractor()
target_shots, baseline_shots = batch_kf.extract_all(target_shots, baseline_shots)

# ── 步骤 4：Embedding（批量，缓存） ───────────────────────────────
batch_emb = BatchEmbedder()
target_emb, baseline_emb = batch_emb.encode_all(
    target_shots, baseline_shots,
    target_video_id="bakemonogatari_s1",
    baseline_video_id="standard_animation",
)

# ── 步骤 5：差异比对 ──────────────────────────────────────────────
diff_engine = DifferentialEngine()
diff_engine.set_shots(target_shots, baseline_shots)
diff_engine.set_embeddings(target_emb, baseline_emb)

pairs = diff_engine.compare(mode="cross", n_pairs=200)

# ── 步骤 6：导出 LLM Prompt ──────────────────────────────────────
diff_engine.export_prompts(pairs, output_name="bakemonogatari_vs_standard")
```

## Data Directory Layout

```
data/
├── raw/
│   ├── target/               # Videos to analyze
│   └── baseline/             # Reference videos for comparison
├── frames/
│   ├── target/               # Extracted keyframes (target)
│   └── baseline/             # Extracted keyframes (baseline)
├── cache/
│   └── embeddings/           # Cached embeddings (.npy)
└── output/
    └── prompts/              # LLM analysis prompts (JSON)
```

## License

MIT
