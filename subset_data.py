import argparse
import json
import csv
from pathlib import Path


def subset(src: Path, dst: Path, doc_frac: float, query_frac: float):
    dst.mkdir(parents=True, exist_ok=True)
    corpus_src = src / "corpus.jsonl"
    queries_src = src / "queries.jsonl"
    qrels_src = src / "qrels" / "test.tsv"

    docs = []
    with corpus_src.open() as f:
        for line in f:
            docs.append(line)
    keep_docs = max(1, int(len(docs) * doc_frac))
    docs = docs[:keep_docs]
    doc_ids = [json.loads(x)["_id"] for x in docs]
    with (dst / "corpus.jsonl").open("w") as f:
        for line in docs:
            f.write(line)

    queries = []
    with queries_src.open() as f:
        for line in f:
            queries.append(line)
    keep_queries = max(1, int(len(queries) * query_frac))
    queries = queries[:keep_queries]
    query_ids = [json.loads(x)["_id"] for x in queries]
    with (dst / "queries.jsonl").open("w") as f:
        for line in queries:
            f.write(line)

    (dst / "qrels").mkdir(exist_ok=True)
    with qrels_src.open() as f_in, (dst / "qrels" / "test.tsv").open("w", newline="") as f_out:
        r = csv.reader(f_in, delimiter="\t")
        w = csv.writer(f_out, delimiter="\t")
        header = next(r, None)
        if header:
            w.writerow(header)
        for qid, docid, rel in r:
            if qid in query_ids and docid in doc_ids:
                w.writerow([qid, docid, rel])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--dest", required=True)
    parser.add_argument("--doc-frac", type=float, default=0.1)
    parser.add_argument("--query-frac", type=float, default=0.1)
    args = parser.parse_args()

    subset(Path(args.source), Path(args.dest), args.doc_frac, args.query_frac)


if __name__ == "__main__":
    main()

