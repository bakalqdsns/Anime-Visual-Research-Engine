-- VSDE PostgreSQL Schema
-- Database: vsde
-- Requires: pgvector extension

-- Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Anime works table (supports multi-work extension)
CREATE TABLE IF NOT EXISTS anime (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    director VARCHAR(255),
    studio VARCHAR(255),
    year INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Shot table
CREATE TABLE IF NOT EXISTS shot (
    id SERIAL PRIMARY KEY,
    shot_id VARCHAR(100) UNIQUE NOT NULL,
    anime_id INTEGER REFERENCES anime(id),
    episode INTEGER,
    video_type VARCHAR(20) DEFAULT 'target' CHECK (video_type IN ('target', 'baseline')),
    start_sec FLOAT,
    end_sec FLOAT,
    duration_sec FLOAT,
    frame_count INTEGER,
    frame_path VARCHAR(500),
    dominant_colors JSONB,
    composition_type VARCHAR(50),
    camera_type VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Embedding storage (pgvector)
CREATE TABLE IF NOT EXISTS shot_embedding (
    shot_id VARCHAR(100) PRIMARY KEY REFERENCES shot(shot_id) ON DELETE CASCADE,
    embedding VECTOR(1792),
    model VARCHAR(50)  -- 'dinov2', 'clip', or 'concat'
);

-- Shot pair difference table
CREATE TABLE IF NOT EXISTS shot_pair (
    id SERIAL PRIMARY KEY,
    shot_a_id VARCHAR(100) REFERENCES shot(shot_id) ON DELETE CASCADE,
    shot_b_id VARCHAR(100) REFERENCES shot(shot_id) ON DELETE CASCADE,
    diff_vector VECTOR(1792),
    diff_magnitude FLOAT,
    llm_analysis JSONB,
    UNIQUE(shot_a_id, shot_b_id)
);

-- Baseline reference table (stores pre-computed center vectors from reference works)
CREATE TABLE IF NOT EXISTS baseline_reference (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,  -- e.g. 'hibike_baseline', 'standard_animation_v1'
    anime_id INTEGER REFERENCES anime(id),
    strategy VARCHAR(30) DEFAULT 'mean',  -- 'mean', 'trimmed_mean', 'pca_center'
    embedding_vector VECTOR(1792),
    shot_count INTEGER,
    embedding_dim INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

COMMENT ON TABLE baseline_reference IS 'Pre-computed center vectors from reference anime works, used as baselines for style difference analysis';

-- Shot diff analysis table (one row per target shot vs baseline)
CREATE TABLE IF NOT EXISTS shot_diff_analysis (
    id SERIAL PRIMARY KEY,
    shot_id VARCHAR(100) REFERENCES shot(shot_id) ON DELETE CASCADE,
    baseline_id INTEGER REFERENCES baseline_reference(id),
    diff_vector VECTOR(1792),
    diff_magnitude FLOAT,
    cosine_similarity FLOAT,
    top_diff_dimensions INTEGER[],
    llm_analysis JSONB,
    UNIQUE(shot_id, baseline_id)
);

COMMENT ON TABLE shot_diff_analysis IS 'Stores per-shot deviation from the baseline center vector (vs_baseline mode)';

-- Style cluster table
CREATE TABLE IF NOT EXISTS style_cluster (
    id SERIAL PRIMARY KEY,
    cluster_label VARCHAR(50),
    embedding_center VECTOR(1792),
    shot_count INTEGER,
    description TEXT
);

-- Cluster assignment table
CREATE TABLE IF NOT EXISTS cluster_assignment (
    id SERIAL PRIMARY KEY,
    shot_id VARCHAR(100) REFERENCES shot(shot_id) ON DELETE CASCADE,
    cluster_id INTEGER REFERENCES style_cluster(id) ON DELETE CASCADE,
    UNIQUE(shot_id)
);

-- Indexes for performance (required after >= 100k rows)
CREATE INDEX IF NOT EXISTS idx_shot_anime ON shot(anime_id);
CREATE INDEX IF NOT EXISTS idx_shot_episode ON shot(episode);
CREATE INDEX IF NOT EXISTS idx_cluster_assignment_shot ON cluster_assignment(shot_id);
CREATE INDEX IF NOT EXISTS idx_cluster_assignment_cluster ON cluster_assignment(cluster_id);

-- HNSW indexes for vector similarity search (build after data is loaded)
-- CREATE INDEX IF NOT EXISTS idx_embedding_hnsw ON shot_embedding USING hnsw (embedding vector_cosine_ops);
-- CREATE INDEX IF NOT EXISTS idx_diff_vector_hnsw ON shot_pair USING hnsw (diff_vector vector_cosine_ops);

COMMENT ON TABLE shot_embedding IS 'Stores 1792-dim DINOv2+CLIP concatenated embeddings per shot';
COMMENT ON TABLE shot_pair IS 'Stores pairwise style difference vectors between shots';
COMMENT ON TABLE style_cluster IS 'Discovered style clusters from HDBSCAN clustering';
