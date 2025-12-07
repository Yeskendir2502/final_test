import argparse
import json
import csv
from pathlib import Path


def subset(src: Path, dst: Path, query_frac: float):
    dst.mkdir(parents=True, exist_ok=True)
    corpus_src = src / "corpus.jsonl"
    queries_src = src / "queries.jsonl"
    qrels_src = src / "qrels" / "test.tsv"

    with queries_src.open() as f:
        queries = [json.loads(line) for line in f]
    with qrels_src.open() as f:
        r = csv.reader(f, delimiter="\t")
        header = next(r, None)
        qrels_rows = [(qid, docid, rel) for qid, docid, rel in r]

    all_qids = [q["_id"] for q in queries if any(qid == q["_id"] for qid, _, _ in qrels_rows)]
    keep_q = max(1, int(len(all_qids) * query_frac))
    keep_qids = set(all_qids[:keep_q])

    keep_docids = set(docid for qid, docid, _ in qrels_rows if qid in keep_qids)

    with corpus_src.open() as f_in, (dst / "corpus.jsonl").open("w") as f_out:
        for line in f_in:
            obj = json.loads(line)
            if obj["_id"] in keep_docids:
                f_out.write(line)

    with (dst / "queries.jsonl").open("w") as f_out:
        for q in queries:
            if q["_id"] in keep_qids:
                f_out.write(json.dumps(q) + "\n")

    (dst / "qrels").mkdir(exist_ok=True)
    with (dst / "qrels" / "test.tsv").open("w", newline="") as f_out:
        w = csv.writer(f_out, delimiter="\t")
        if header:
            w.writerow(header)
        for qid, docid, rel in qrels_rows:
            if qid in keep_qids:
                w.writerow([qid, docid, rel])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--dest", required=True)
    parser.add_argument("--query-frac", type=float, default=0.1)
    args = parser.parse_args()

    subset(Path(args.source), Path(args.dest), args.query_frac)


if __name__ == "__main__":
    main()

