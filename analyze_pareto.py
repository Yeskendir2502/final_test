import json
from pathlib import Path


def summarize_pareto(path: Path):
    data = json.loads(path.read_text())
    fitness = data.get("fitness", [])
    if not fitness:
        return None
    best_idx = min(range(len(fitness)), key=lambda i: fitness[i][0])  # f1 = -ndcg
    f1, f2 = fitness[best_idx]
    ndcg = -f1
    latency_ms = f2
    return {
        "file": str(path),
        "pareto_size": len(fitness),
        "best_ndcg": ndcg,
        "latency_ms": latency_ms,
    }


def main():
    artifacts = Path("artifacts")
    reports = []
    for p in artifacts.glob("nsga2_*.json"):
        r = summarize_pareto(p)
        if r:
            reports.append(r)

    if not reports:
        print("No Pareto files found in artifacts/nsga2_*.json")
        return

    for r in reports:
        print(
            f"{Path(r['file']).name}: pareto_size={r['pareto_size']} "
            f"best_ndcg={r['best_ndcg']:.4f} latency_ms={r['latency_ms']:.2f}"
        )


if __name__ == "__main__":
    main()

