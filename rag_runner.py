import json
import math
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm

from GA_BOHB.optimizer.optimizer import ConfigOptimizer
from NSGA2.nsga2_core import run_nsga2
from NSGA2.representation import decode_chromosome
from rag_pipeline.pipeline import evaluate_config

# hardcoded params (style inherited from just_example.py)
MY_SEED = 42
POP_COUNT = 24
GENERATIONS = 6
RANDOM_TRIALS = 8
CROSS_PROB = 0.9
MUT_PROB = 0.15
DATASETS = ["fiqa", "scifact"]
EMBED_MODELS = ["all-MiniLM-L6-v2", "bge-base", "mpnet"]
USE_DUMMY = True  # set False on the Linux server for real metrics


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def flatten_cfg(cfg):
    clean = {}
    for k, v in cfg.items():
        if isinstance(v, list):
            clean[k] = v[0]
        else:
            clean[k] = v
    return clean


def run_nsga(dataset: str):
    def eval_fn(chromosome):
        cfg = flatten_cfg(decode_chromosome(chromosome))
        cfg["embedding_model"] = random.choice(EMBED_MODELS)
        res = evaluate_config(cfg, dataset=dataset, use_dummy=USE_DUMMY, return_details=True)
        return (-res["ndcg"], res["latency_ms"])

    pareto, pf_fit, _, _ = run_nsga2(
        eval_fn=eval_fn,
        pop_size=POP_COUNT,
        n_generations=GENERATIONS,
        crossover_prob=CROSS_PROB,
        mutation_prob=MUT_PROB,
        seed=MY_SEED,
    )
    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / f"nsga2_{dataset}.json", "w", encoding="utf-8") as f:
        json.dump({"pareto": pareto, "fitness": pf_fit}, f, indent=2)


def run_random(dataset: str):
    opt = ConfigOptimizer(dataset=dataset, use_dummy=USE_DUMMY)
    opt.run_random_search(trials=RANDOM_TRIALS, dataset=dataset, use_dummy=USE_DUMMY)


def run_grid(dataset: str):
    base_cfg = {
        "splitter_type": "token",
        "chunk_size": 256,
        "chunk_overlap": 0,
        "normalize_embeddings": True,
        "index_type": "HNSW",
        "hnsw_M": 24,
        "hnsw_efSearch": 100,
        "top_k": 10,
        "hybrid_weight": 0.5,
    }
    history = Path("artifacts/history_grid.jsonl")
    history.parent.mkdir(exist_ok=True)
    configs = []
    for model in EMBED_MODELS:
        cfg = dict(base_cfg)
        cfg["embedding_model"] = model
        configs.append(cfg)

    for cfg in tqdm(configs, desc=f"grid-{dataset}"):
        res = evaluate_config(cfg, dataset=dataset, use_dummy=USE_DUMMY, return_details=True)
        with open(history, "a", encoding="utf-8") as f:
            f.write(json.dumps({"dataset": dataset, "config": cfg, **res}) + "\n")


def main():
    set_seeds(MY_SEED)
    for ds in DATASETS:
        print(f"=== DATASET {ds} :: grid ===")
        run_grid(ds)
        print(f"=== DATASET {ds} :: random ===")
        run_random(ds)
        print(f"=== DATASET {ds} :: nsga2 ===")
        run_nsga(ds)
    print("Done.")


if __name__ == "__main__":
    main()

