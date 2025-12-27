from __future__ import annotations

import argparse
from pathlib import Path

from fastfractal.core.decode import decode_to_file
from fastfractal.core.encode import encode_to_file
from fastfractal.io.config import (
    get_section,
    load_yaml,
    pick_bool,
    pick_float,
    pick_int,
    pick_opt_int,
    pick_str,
)


def main() -> None:
    p = argparse.ArgumentParser(prog="fastfractal")
    sp = p.add_subparsers(dest="cmd", required=True)

    pe = sp.add_parser("encode")
    pe.add_argument("input", type=Path)
    pe.add_argument("output", type=Path)
    pe.add_argument("--config", type=Path, default=None)

    pe.add_argument("--max-block", type=int, default=None)
    pe.add_argument("--min-block", type=int, default=None)
    pe.add_argument("--stride", type=int, default=None)

    gq = pe.add_mutually_exclusive_group()
    gq.add_argument(
        "--quadtree", dest="quadtree", action="store_const", const=True, default=None
    )
    gq.add_argument("--no-quadtree", dest="quadtree", action="store_const", const=False)

    pe.add_argument("--max-mse", type=float, default=None)

    pe.add_argument("--buckets", type=int, default=None)

    gs = pe.add_mutually_exclusive_group()
    gs.add_argument(
        "--s-sets", dest="s_sets", action="store_const", const=True, default=None
    )
    gs.add_argument("--no-s-sets", dest="s_sets", action="store_const", const=False)

    gk = pe.add_mutually_exclusive_group()
    gk.add_argument(
        "--quantize", dest="quantize", action="store_const", const=True, default=None
    )
    gk.add_argument("--no-quantize", dest="quantize", action="store_const", const=False)

    pe.add_argument("--s-clip", type=float, default=None)
    pe.add_argument("--o-min", type=float, default=None)
    pe.add_argument("--o-max", type=float, default=None)

    pe.add_argument(
        "--backend", type=str, default=None, choices=["dot", "lsh", "pca_lsh"]
    )
    pe.add_argument("--topk", type=int, default=None)
    pe.add_argument("--lsh-budget", type=int, default=None)
    pe.add_argument("--pca-dim", type=int, default=None)
    pe.add_argument("--lsh-planes", type=int, default=None)
    pe.add_argument("--seed", type=int, default=None)

    pe.add_argument("--entropy-thresh", type=float, default=None)
    pe.add_argument("--max-domains", type=int, default=None)

    pd = sp.add_parser("decode")
    pd.add_argument("input", type=Path)
    pd.add_argument("output", type=Path)
    pd.add_argument("--config", type=Path, default=None)
    pd.add_argument("--iters", type=int, default=None)

    a = p.parse_args()

    cfg_all: dict[str, object] = {}
    if getattr(a, "config", None) is not None:
        cfg_all = load_yaml(a.config)

    if a.cmd == "encode":
        cfg = get_section(cfg_all, "encode")

        max_block = pick_int(cfg, "max_block", a.max_block, 16)
        min_block = pick_int(cfg, "min_block", a.min_block, 4)
        stride = pick_int(cfg, "stride", a.stride, 4)

        use_quadtree = pick_bool(cfg, "quadtree", a.quadtree, False)
        max_mse = pick_float(cfg, "max_mse", a.max_mse, 0.0025)

        buckets = pick_int(cfg, "buckets", a.buckets, 0)
        use_buckets = buckets > 0
        bucket_count = buckets if use_buckets else 1

        use_s_sets = pick_bool(cfg, "s_sets", a.s_sets, False)

        quantized = pick_bool(cfg, "quantize", a.quantize, False)
        s_clip = pick_float(cfg, "s_clip", a.s_clip, 0.99)
        o_min = pick_float(cfg, "o_min", a.o_min, -0.5)
        o_max = pick_float(cfg, "o_max", a.o_max, 1.5)

        backend = pick_str(cfg, "backend", a.backend, "dot")
        topk = pick_int(cfg, "topk", a.topk, 64)
        lsh_budget = pick_int(cfg, "lsh_budget", a.lsh_budget, 2048)
        pca_dim = pick_int(cfg, "pca_dim", a.pca_dim, 16)
        lsh_planes = pick_int(cfg, "lsh_planes", a.lsh_planes, 16)
        seed = pick_int(cfg, "seed", a.seed, 0)

        entropy_thresh = pick_float(cfg, "entropy_thresh", a.entropy_thresh, 0.0)
        max_domains = pick_opt_int(cfg, "max_domains", a.max_domains, None)

        encode_to_file(
            a.input,
            a.output,
            min_block=min_block,
            max_block=max_block,
            stride=stride,
            use_quadtree=use_quadtree,
            max_mse=max_mse,
            use_buckets=use_buckets,
            bucket_count=bucket_count,
            use_s_sets=use_s_sets,
            topk=topk,
            backend=backend,
            lsh_budget=lsh_budget,
            entropy_thresh=entropy_thresh,
            quantized=quantized,
            s_clip=s_clip,
            o_min=o_min,
            o_max=o_max,
            pca_dim=pca_dim,
            lsh_planes=lsh_planes,
            seed=seed,
            max_domains=max_domains,
        )
        return

    cfg = get_section(cfg_all, "decode")
    iters = pick_int(cfg, "iters", a.iters, 8)
    decode_to_file(a.input, a.output, iterations=iters)


if __name__ == "__main__":
    main()
