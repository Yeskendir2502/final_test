import shutil
from pathlib import Path

from rag_pipeline.pipeline import evaluate_config


def test_evaluate_config_dummy_runs_and_caches(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    cfg = {
        "splitter_type": "token",
        "chunk_size": 256,
        "chunk_overlap": 0,
        "embedding_model": "all-MiniLM-L6-v2",
        "normalize_embeddings": True,
        "index_type": "HNSW",
        "hnsw_M": 16,
        "hnsw_efSearch": 50,
        "top_k": 5,
        "hybrid_weight": 0.0,
    }

    # First run builds cache.
    result1 = evaluate_config(cfg, dataset="fiqa", use_dummy=True, cache_dir=cache_dir, return_details=True)
    assert "ndcg" in result1 and "latency_ms" in result1
    assert result1["cache_hit"] is False

    # Modify only retrieval params -> should reuse cache.
    cfg2 = dict(cfg)
    cfg2["top_k"] = 10
    result2 = evaluate_config(cfg2, dataset="fiqa", use_dummy=True, cache_dir=cache_dir, return_details=True)
    assert result2["cache_hit"] is True

    # Clean up
    shutil.rmtree(cache_dir, ignore_errors=True)

