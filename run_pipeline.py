import argparse
import json
from pathlib import Path
from typing import Dict

from tqdm import tqdm

from GA_BOHB.optimizer.optimizer import ConfigOptimizer
from NSGA2.nsga2_core import run_nsga2
from NSGA2.representation import decode_chromosome
from rag_pipeline.pipeline import evaluate_config


def _flatten_config(config_with_lists: Dict[str, list | int | float | str | bool]) -> Dict[str, object]:
    clean = {}
    for k, v in config_with_lists.items():
        if isinstance(v, list):
            clean[k] = v[0]
        else:
            clean[k] = v
    return clean


def run_bohb(args):
    opt = ConfigOptimizer(dataset=args.dataset, use_dummy=args.use_dummy)
    if args.initial_configs:
        opt.run_initial_configs(dataset=args.dataset, use_dummy=args.use_dummy)
    if args.trials > 0:
        opt.run_random_search(trials=args.trials, dataset=args.dataset, use_dummy=args.use_dummy)


def run_nsga(args):
    def eval_fn(chromosome):
        cfg = _flatten_config(decode_chromosome(chromosome))
        result = evaluate_config(cfg, dataset=args.dataset, use_dummy=args.use_dummy, return_details=True)
        # NSGA-II minimizes; return negative ndcg for maximization.
        return (-result["ndcg"], result["latency_ms"])

    pareto, pf_fit, _, _ = run_nsga2(
        eval_fn=eval_fn,
        pop_size=args.population,
        n_generations=args.generations,
        crossover_prob=args.crossover,
        mutation_prob=args.mutation,
        seed=args.seed,
    )
    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "nsga2_pareto.json", "w", encoding="utf-8") as f:
        json.dump({"pareto": pareto, "fitness": pf_fit}, f, indent=2)


def run_grid(args):
    # Simple loop to show tqdm progress and logging of results.
    configs = [
        {"splitter_type": "token", "chunk_size": 256, "chunk_overlap": 0, "embedding_model": "all-MiniLM-L6-v2",
         "normalize_embeddings": True, "index_type": "HNSW", "hnsw_M": 24, "hnsw_efSearch": 50, "top_k": 5,
         "hybrid_weight": 0.0},
        {"splitter_type": "sentence", "chunk_size": 384, "chunk_overlap": 30, "embedding_model": "bge-base",
         "normalize_embeddings": True, "index_type": "HNSW", "hnsw_M": 24, "hnsw_efSearch": 100, "top_k": 10,
         "hybrid_weight": 0.5},
    ]
    history_path = Path("artifacts/history.jsonl")
    history_path.parent.mkdir(exist_ok=True)
    for cfg in tqdm(configs, desc="Grid", unit="config"):
        result = evaluate_config(cfg, dataset=args.dataset, use_dummy=args.use_dummy, return_details=True)
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"config": cfg, **result}) + "\n")


def main():
    parser = argparse.ArgumentParser(description="RAG pipeline optimizer runner")
    parser.add_argument("--dataset", choices=["fiqa", "scifact"], default="fiqa")
    parser.add_argument("--algo", choices=["bohb", "nsga2", "grid"], default="bohb")
    parser.add_argument("--trials", type=int, default=5, help="Trials for BOHB/random search")
    parser.add_argument("--initial-configs", action="store_true", help="Run initial configs before search")
    parser.add_argument("--generations", type=int, default=5, help="Generations for NSGA-II")
    parser.add_argument("--population", type=int, default=12, help="Population size for NSGA-II")
    parser.add_argument("--crossover", type=float, default=0.9)
    parser.add_argument("--mutation", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-dummy", action="store_true", help="Use lightweight dummy evaluator (fast tests)")
    args = parser.parse_args()

    if args.algo == "bohb":
        run_bohb(args)
    elif args.algo == "nsga2":
        run_nsga(args)
    else:
        run_grid(args)


if __name__ == "__main__":
    main()

