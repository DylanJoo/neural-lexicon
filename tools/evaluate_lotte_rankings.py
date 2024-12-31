import argparse
from collections import defaultdict
import jsonlines
import os


def evaluate_dataset(query_type, dataset, split, retrieval, k, data_rootdir, rankings_rootdir):
    data_path = os.path.join(data_rootdir, dataset, split)

    # [modify] change the path into trec-style runs

    # ---- modify start ----
    rankings_path = os.path.join(
        rankings_rootdir, 
        f"run.lotte.{dataset}.{retrieval}.txt"
    )
    print(rankings_path)
    if not os.path.exists(rankings_path):
        print(f"[query_type={query_type}, dataset={dataset}] Success@{k}: ???")
        return
    rankings = defaultdict(list)
    with open(rankings_path, "r") as f:
        for line in f:
            items = line.strip().split()
            qid, _, pid, rank = items[:4]
            qid = int(qid)
            pid = int(pid)
            rank = int(rank)
            rankings[qid].append(pid)
            assert rank == len(rankings[qid])
    # ---- modify start ----

    success = 0
    qas_path = os.path.join(data_path, f"qas.{query_type}.jsonl")

    with jsonlines.open(qas_path, mode="r") as f:
        for line in f:
            qid = int(line["qid"])
            answer_pids = set(line["answer_pids"])
            if len(set(rankings[qid][:k]).intersection(answer_pids)) > 0:
                success += 1
    print(
        f"[query_type={query_type}, dataset={dataset}] "
        f"Success@{k}: {success / len(rankings) * 100:.1f}"
    )


def main(args):
    # for query_type in ["search", "forum"]:
    for query_type in ["search"]:
        for dataset in [
            "writing",
            "recreation",
            "science",
            "technology",
            "lifestyle",
            # "pooled", # [NOTE] we didnt use pooled here
        ]:
            evaluate_dataset(
                query_type,
                dataset,
                args.split,
                args.retrieval,
                args.k,
                args.data_dir,
                args.rankings_dir,
            )
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoTTE evaluation script")
    parser.add_argument("--k", type=int, default=5, help="Success@k")
    parser.add_argument("-s", "--split", choices=["dev", "test"], required=True, help="Split")
    parser.add_argument("-d", "--data_dir", type=str, required=True, help="Path to LoTTE data directory")
    parser.add_argument("--retrieval", type=str, default='bm25')
    parser.add_argument(
        "-r",
        "--rankings_dir",
        type=str,
        required=True,
        help="Path to LoTTE rankings directory",
    )
    args = parser.parse_args()
    main(args)
