#!/usr/bin/env python3
import argparse, random
from pathlib import Path

def node_name(i, width):
    return f"n{str(i).zfill(width)}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=2000, help="number of graph nodes")
    ap.add_argument("--avg-outdeg", type=float, default=4.0, help="average out-degree")
    ap.add_argument("--queries", type=int, default=100, help="number of ground queries")
    ap.add_argument("--seed", type=int, default=12345, help="random seed (for repeatability)")
    ap.add_argument("--out", type=Path, default=Path("programs/benchmark.dl"))
    args = ap.parse_args()

    random.seed(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    width = len(str(args.nodes - 1))

    # Generate edges
    edges = set()
    for u in range(args.nodes):
        k = max(0, int(random.gauss(args.avg_outdeg, args.avg_outdeg**0.5)))
        for _ in range(k):
            v = random.randrange(args.nodes)
            if u != v:
                edges.add((u, v))

    with open(args.out, "w") as f:
        # Facts
        f.write("% --- Facts ---\n")
        for u, v in sorted(edges):
            f.write(f"edge({node_name(u,width)}, {node_name(v,width)}).\n")

        # Rules
        f.write("\n% --- Rules ---\n")
        f.write("path(X, Y) :- edge(X, Y).\n")
        f.write("path(X, Z) :- edge(X, Y), path(Y, Z).\n")

        # Queries
        f.write("\n% --- Queries ---\n")
        # Some deterministic queries
        f.write("?- path(X, Y).\n")   # dump everything
        # Some ground true/false queries
        nodes = [node_name(i, width) for i in range(args.nodes)]
        for _ in range(args.queries):
            a, b = random.sample(nodes, 2)
            f.write(f"?- path({a}, {b}).\n")

    print(f"Wrote benchmark program to {args.out} "
          f"with {len(edges)} edges and {args.queries+1} queries.")

if __name__ == "__main__":
    main()
