from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from fastfractal import _cext  # type: ignore
from fastfractal.core.blocks import extract_range
from fastfractal.core.poolbuilder import bucket_id, build_pool
from fastfractal.core.quantization import (
    _clipf,
    dequant_o,
    dequant_s,
    quant_o,
    quant_s,
    rgb_to_luma,
)
from fastfractal.core.regression import RegressionKind, resolve_regression
from fastfractal.core.search import (
    SearchBackend,
    topk_from_subset,
)
from fastfractal.core.types import FractalCode, PoolRuntime
from fastfractal.io.codebook import save_code
from fastfractal.io.imageio import load_image
from fastfractal.utils.entropy import entropy01, var01

if TYPE_CHECKING:
    from numpy.typing import NDArray

_HAS_TOPK = hasattr(_cext, "topk_from_subset")
_HAS_ENCODE_LEAF = hasattr(_cext, "encode_leaf_best")


def pad_to_multiple(
    img: NDArray[np.float32], block: int
) -> tuple[NDArray[np.float32], int, int]:
    h = int(img.shape[0])
    w = int(img.shape[1])
    ph = (block - (h % block)) % block
    pw = (block - (w % block)) % block
    if ph == 0 and pw == 0:
        return img, h, w
    if img.ndim == 2:
        out = np.pad(img, ((0, ph), (0, pw)), mode="edge").astype(
            np.float32, copy=False
        )
    else:
        out = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="edge").astype(
            np.float32, copy=False
        )
    return out, h, w


def default_s_sets(bucket_count: int, s_clip: float) -> list[list[float]]:
    out: list[list[float]] = []
    for i in range(bucket_count):
        t = 0.0 if bucket_count <= 1 else float(i) / float(bucket_count - 1)
        mx = 0.2 + 0.79 * t
        mx = _clipf(mx, 0.05, float(s_clip))
        vals = np.linspace(0.0, mx, 5, dtype=np.float64)
        sset = [float(v) for v in vals]
        if i >= bucket_count // 2:
            neg = [-float(v) for v in vals[1:3]]
            sset = neg + sset
        out.append(sset)
    return out


def choose_s_from_set(s: float, sset: list[float]) -> float:
    if not sset:
        return s
    best = sset[0]
    bd = abs(best - s)
    for v in sset[1:]:
        d = abs(v - s)
        if d < bd:
            bd = d
            best = v
    return best


def pool_query_candidates(
    pool: PoolRuntime,
    q: NDArray[np.float32],
    bid: int,
    topk: int,
    lsh_budget: int,
) -> NDArray[np.int64]:
    if pool.backend == "dot":
        subset = pool.bucket_entries[bid]
        if _HAS_TOPK:
            return _cext.topk_from_subset(pool.proxy_mat, q, subset, int(topk))  # type: ignore[no-any-return]
        return topk_from_subset(pool.proxy_mat, q, subset, topk)

    if pool.backend == "lsh":
        if pool.lsh is None:
            raise ValueError("missing lsh")
        cand = pool.lsh.query(q, budget=lsh_budget)
        if cand.size == 0:
            subset = pool.bucket_entries[bid]
            return topk_from_subset(pool.proxy_mat, q, subset, topk)
        if len(pool.bucket_entries) > 1:
            m = pool.entry_bucket[cand] == np.uint8(bid)
            cand2 = cand[m]
            if cand2.size == 0:
                cand2 = cand
        else:
            cand2 = cand
        return topk_from_subset(
            pool.proxy_mat, q, cand2.astype(np.int64, copy=False), topk
        )

    if pool.backend == "pca_lsh":
        if pool.lsh is None or pool.pca_mean is None or pool.pca_basis is None:
            raise ValueError("missing pca/lsh")
        q2 = (q - pool.pca_mean) @ pool.pca_basis.T
        cand = pool.lsh.query(q2.astype(np.float32, copy=False), budget=lsh_budget)
        if cand.size == 0:
            subset = pool.bucket_entries[bid]
            return topk_from_subset(pool.proxy_mat, q, subset, topk)
        if len(pool.bucket_entries) > 1:
            m = pool.entry_bucket[cand] == np.uint8(bid)
            cand2 = cand[m]
            if cand2.size == 0:
                cand2 = cand
        else:
            cand2 = cand
        return topk_from_subset(
            pool.proxy_mat, q, cand2.astype(np.int64, copy=False), topk
        )

    raise ValueError("bad backend")


def encode_leaf(
    img: NDArray[np.float32],
    luma: NDArray[np.float32],
    pool: PoolRuntime,
    y: int,
    x: int,
    bucket_count: int,
    use_buckets: bool,
    use_s_sets: bool,
    s_sets: list[list[float]],
    s_clip: float,
    o_min: float,
    o_max: float,
    quantized: bool,
    topk: int,
    lsh_budget: int,
    regression: RegressionKind | str = "linear",
) -> tuple[int, int, NDArray[np.uint8] | NDArray[np.float32], float]:
    b = int(pool.block)
    c = 1 if img.ndim == 2 else int(img.shape[2])
    n_pix = int(b * b)
    n = float(n_pix)
    inv_n = 1.0 / n

    bid = 0
    if use_buckets:
        rproxy2 = extract_range(luma, y, x, b).astype(np.float32, copy=False)
        ent = float(entropy01(rproxy2))
        vr = var01(rproxy2)
        bid = bucket_id(ent, vr, int(bucket_count))

    rproxy = extract_range(luma, y, x, b).ravel().astype(np.float32, copy=False)
    nrm2 = float(np.dot(rproxy, rproxy))
    if nrm2 > 1e-24:
        inv = 1.0 / math.sqrt(nrm2)
        q = (rproxy * np.float32(inv)).astype(np.float32, copy=False)
    else:
        q = rproxy

    cand = pool_query_candidates(
        pool, q, bid, topk=int(topk), lsh_budget=int(lsh_budget)
    )

    reg_fn = resolve_regression(regression)

    if _HAS_ENCODE_LEAF and not (use_s_sets and use_buckets):
        cand_i32 = np.ascontiguousarray(cand, dtype=np.int32)

        return _cext.encode_leaf_best(  # type: ignore[attr-defined, no-any-return, unused-ignore]
            img,
            pool.tf_flat,
            pool.tf_sum if pool.tf_sum is not None else None,
            pool.tf_sum2 if pool.tf_sum2 is not None else None,
            pool.map_dom,
            pool.map_tf,
            int(y),
            int(x),
            int(b),
            cand_i32,
            float(s_clip),
            float(o_min),
            float(o_max),
            int(quantized),
            str(regression),
        )

    if quantized:
        best_codes_q = np.zeros((c, 2), dtype=np.uint8)
    else:
        best_codes_f = np.zeros((c, 2), dtype=np.float32)

    best_mse = float("inf")
    best_dom = 0
    best_tf = 0

    def post_s_scalar(s0: float) -> float:
        if use_s_sets and use_buckets:
            s0 = choose_s_from_set(s0, s_sets[bid])
        return _clipf(s0, -float(s_clip), float(s_clip))

    tf_sum = pool.tf_sum
    tf_sum2 = pool.tf_sum2

    if c == 1:
        r = extract_range(img, y, x, b).ravel().astype(np.float32, copy=False)
        sumR = float(r.sum(dtype=np.float64))
        sumRR = float(np.dot(r, r))

        for ci in cand:
            k = int(ci)
            domv = pool.tf_flat[k, 0, :]

            s0, o0, _ = reg_fn(domv, r)
            s1 = post_s_scalar(s0)
            o1 = _clipf(o0, float(o_min), float(o_max))

            if tf_sum is not None and tf_sum2 is not None:
                sumD = float(tf_sum[k, 0])
                sumDD = float(tf_sum2[k, 0])
            else:
                sumD = float(domv.sum(dtype=np.float64))
                sumDD = float(np.dot(domv, domv))

            sumRD = float(np.dot(domv, r))

            if quantized:
                qs = quant_s(s1, s_clip)
                qo = quant_o(o1, o_min, o_max)
                s2 = dequant_s(qs, s_clip)
                o2 = dequant_o(qo, o_min, o_max)
                sse = (
                    sumRR
                    + (s2 * s2) * sumDD
                    + n * (o2 * o2)
                    - 2.0 * s2 * sumRD
                    - 2.0 * o2 * sumR
                    + 2.0 * s2 * o2 * sumD
                )
                mse = float(sse * inv_n)
                if mse < best_mse:
                    best_mse = mse
                    best_dom = int(pool.map_dom[k])
                    best_tf = int(pool.map_tf[k])
                    best_codes_q[0, 0] = np.uint8(qs)
                    best_codes_q[0, 1] = np.uint8(qo)
            else:
                sse = (
                    sumRR
                    + (s1 * s1) * sumDD
                    + n * (o1 * o1)
                    - 2.0 * s1 * sumRD
                    - 2.0 * o1 * sumR
                    + 2.0 * s1 * o1 * sumD
                )
                mse = float(sse * inv_n)
                if mse < best_mse:
                    best_mse = mse
                    best_dom = int(pool.map_dom[k])
                    best_tf = int(pool.map_tf[k])
                    best_codes_f[0, 0] = np.float32(s1)
                    best_codes_f[0, 1] = np.float32(o1)

        if quantized:
            return best_dom, best_tf, best_codes_q, best_mse
        return best_dom, best_tf, best_codes_f, best_mse

    rblk = img[y : y + b, x : x + b, :].astype(np.float32, copy=False)
    rflat = np.transpose(rblk, (2, 0, 1)).reshape(c, -1).astype(np.float32, copy=False)

    sumR = rflat.sum(axis=1, dtype=np.float64)
    sumRR = np.einsum("ij,ij->i", rflat, rflat, dtype=np.float64)

    for ci in cand:
        k = int(ci)
        dom_all = pool.tf_flat[k, :, :]  # (3, n_pix)

        if tf_sum is not None and tf_sum2 is not None:
            sumD0, sumD1, sumD2 = (
                float(tf_sum[k, 0]),
                float(tf_sum[k, 1]),
                float(tf_sum[k, 2]),
            )
            sumDD0, sumDD1, sumDD2 = (
                float(tf_sum2[k, 0]),
                float(tf_sum2[k, 1]),
                float(tf_sum2[k, 2]),
            )
        else:
            sumD0 = float(dom_all[0].sum(dtype=np.float64))
            sumD1 = float(dom_all[1].sum(dtype=np.float64))
            sumD2 = float(dom_all[2].sum(dtype=np.float64))
            sumDD0 = float(np.dot(dom_all[0], dom_all[0]))
            sumDD1 = float(np.dot(dom_all[1], dom_all[1]))
            sumDD2 = float(np.dot(dom_all[2], dom_all[2]))

        sumRD0 = float(np.dot(dom_all[0], rflat[0]))
        sumRD1 = float(np.dot(dom_all[1], rflat[1]))
        sumRD2 = float(np.dot(dom_all[2], rflat[2]))

        def solve(
            sumD: float, sumDD: float, sumRch: float, sumRD: float
        ) -> tuple[float, float]:
            denom = n * sumDD - sumD * sumD
            if abs(denom) < 1e-18:
                return 0.0, (sumRch / n)
            s0 = (n * sumRD - sumD * sumRch) / denom
            o0 = (sumRch - s0 * sumD) / n
            return s0, o0

        s0_0, o0_0 = solve(sumD0, sumDD0, float(sumR[0]), sumRD0)
        s0_1, o0_1 = solve(sumD1, sumDD1, float(sumR[1]), sumRD1)
        s0_2, o0_2 = solve(sumD2, sumDD2, float(sumR[2]), sumRD2)

        s1_0 = post_s_scalar(s0_0)
        s1_1 = post_s_scalar(s0_1)
        s1_2 = post_s_scalar(s0_2)
        o1_0 = _clipf(o0_0, float(o_min), float(o_max))
        o1_1 = _clipf(o0_1, float(o_min), float(o_max))
        o1_2 = _clipf(o0_2, float(o_min), float(o_max))

        if quantized:
            qs0, qo0 = quant_s(s1_0, s_clip), quant_o(o1_0, o_min, o_max)
            qs1, qo1 = quant_s(s1_1, s_clip), quant_o(o1_1, o_min, o_max)
            qs2, qo2 = quant_s(s1_2, s_clip), quant_o(o1_2, o_min, o_max)
            s2_0, o2_0 = dequant_s(qs0, s_clip), dequant_o(qo0, o_min, o_max)
            s2_1, o2_1 = dequant_s(qs1, s_clip), dequant_o(qo1, o_min, o_max)
            s2_2, o2_2 = dequant_s(qs2, s_clip), dequant_o(qo2, o_min, o_max)

            sse0 = (
                float(sumRR[0])
                + (s2_0 * s2_0) * sumDD0
                + n * (o2_0 * o2_0)
                - 2.0 * s2_0 * sumRD0
                - 2.0 * o2_0 * float(sumR[0])
                + 2.0 * s2_0 * o2_0 * sumD0
            )
            sse1 = (
                float(sumRR[1])
                + (s2_1 * s2_1) * sumDD1
                + n * (o2_1 * o2_1)
                - 2.0 * s2_1 * sumRD1
                - 2.0 * o2_1 * float(sumR[1])
                + 2.0 * s2_1 * o2_1 * sumD1
            )
            sse2 = (
                float(sumRR[2])
                + (s2_2 * s2_2) * sumDD2
                + n * (o2_2 * o2_2)
                - 2.0 * s2_2 * sumRD2
                - 2.0 * o2_2 * float(sumR[2])
                + 2.0 * s2_2 * o2_2 * sumD2
            )
            mse = float((sse0 + sse1 + sse2) / (3.0 * n))
            if mse < best_mse:
                best_mse = mse
                best_dom = int(pool.map_dom[k])
                best_tf = int(pool.map_tf[k])
                best_codes_q[:, 0] = np.asarray([qs0, qs1, qs2], dtype=np.uint8)
                best_codes_q[:, 1] = np.asarray([qo0, qo1, qo2], dtype=np.uint8)
        else:
            sse0 = (
                float(sumRR[0])
                + (s1_0 * s1_0) * sumDD0
                + n * (o1_0 * o1_0)
                - 2.0 * s1_0 * sumRD0
                - 2.0 * o1_0 * float(sumR[0])
                + 2.0 * s1_0 * o1_0 * sumD0
            )
            sse1 = (
                float(sumRR[1])
                + (s1_1 * s1_1) * sumDD1
                + n * (o1_1 * o1_1)
                - 2.0 * s1_1 * sumRD1
                - 2.0 * o1_1 * float(sumR[1])
                + 2.0 * s1_1 * o1_1 * sumD1
            )
            sse2 = (
                float(sumRR[2])
                + (s1_2 * s1_2) * sumDD2
                + n * (o1_2 * o1_2)
                - 2.0 * s1_2 * sumRD2
                - 2.0 * o1_2 * float(sumR[2])
                + 2.0 * s1_2 * o1_2 * sumD2
            )
            mse = float((sse0 + sse1 + sse2) / (3.0 * n))
            if mse < best_mse:
                best_mse = mse
                best_dom = int(pool.map_dom[k])
                best_tf = int(pool.map_tf[k])
                best_codes_f[:, 0] = np.asarray([s1_0, s1_1, s1_2], dtype=np.float32)
                best_codes_f[:, 1] = np.asarray([o1_0, o1_1, o1_2], dtype=np.float32)

    if quantized:
        return best_dom, best_tf, best_codes_q, best_mse
    return best_dom, best_tf, best_codes_f, best_mse


def encode_array(
    img: NDArray[np.float32],
    min_block: int = 4,
    max_block: int = 16,
    stride: int = 4,
    use_quadtree: bool = False,
    max_mse: float = 0.0025,
    use_buckets: bool = False,
    bucket_count: int = 8,
    use_s_sets: bool = False,
    topk: int = 64,
    backend: str = "dot",
    transform_ids: tuple[int, ...] | None = None,
    lsh_budget: int = 2048,
    entropy_thresh: float = 0.0,
    quantized: bool = False,
    s_clip: float = 0.99,
    o_min: float = -0.5,
    o_max: float = 1.5,
    pca_dim: int = 16,
    lsh_planes: int = 16,
    seed: int = 0,
    max_domains: int | None = None,
    block: int | None = None,
    iterations_hint: int = 8,
    regression: str | RegressionKind = "linear",
    *,
    precompute_stats: bool = True,
) -> FractalCode:
    if img.ndim not in (2, 3):
        raise ValueError("img must be HxW or HxWxC")

    if block is not None:
        b = int(block)
        if b <= 0:
            raise ValueError("block must be positive")
        min_block = b
        max_block = b
        use_quadtree = False

    orig_h = int(img.shape[0])
    orig_w = int(img.shape[1])
    img2, _, _ = pad_to_multiple(img, max_block)
    h = int(img2.shape[0])
    w = int(img2.shape[1])

    c = 1 if img2.ndim == 2 else int(img2.shape[2])
    if c not in (1, 3):
        raise ValueError("channels must be 1 or 3")

    if min_block <= 0 or max_block <= 0 or (max_block % min_block) != 0:
        raise ValueError("bad blocks")
    if (max_block & (max_block - 1)) != 0 or (min_block & (min_block - 1)) != 0:
        raise ValueError("blocks must be powers of two")
    if backend not in SearchBackend:
        raise ValueError("bad backend")

    if not use_quadtree:
        min_block = max_block

    if not use_buckets:
        bucket_count = 1
        use_s_sets = False

    if use_buckets and bucket_count < 2:
        bucket_count = 2

    s_sets = (
        default_s_sets(bucket_count, s_clip)
        if use_s_sets
        else [[] for _ in range(bucket_count)]
    )

    luma = rgb_to_luma(img2)

    blocks: list[int] = []
    bcur = max_block
    while bcur >= min_block:
        blocks.append(bcur)
        if bcur == min_block:
            break
        bcur //= 2

    pools: list[PoolRuntime] = []
    for bcur in blocks:
        pools.append(
            build_pool(
                img=img2,
                block=bcur,
                stride=stride,
                entropy_thresh=entropy_thresh,
                bucket_count=bucket_count,
                use_buckets=use_buckets,
                backend=backend,
                pca_dim=pca_dim,
                lsh_planes=lsh_planes,
                seed=seed + bcur,
                max_domains=max_domains,
                transform_ids=transform_ids,
                luma=luma,
                precompute_stats=precompute_stats,
            )
        )

    if len(pools) > 255:
        raise ValueError("too many pools")

    pool_idx: dict[int, int] = {p.block: i for i, p in enumerate(pools)}

    leaf_yx_list: list[tuple[int, int]] = []
    leaf_pool_list: list[int] = []
    leaf_dom_list: list[int] = []
    leaf_tf_list: list[int] = []
    leaf_codes_q_list: list[NDArray[np.uint8]] = []
    leaf_codes_f_list: list[NDArray[np.float32]] = []

    def emit_leaf(y0: int, x0: int, block0: int) -> float:
        pi = pool_idx[block0]
        pool = pools[pi]
        dom, tf, codes, mse = encode_leaf(
            img=img2,
            luma=luma,
            pool=pool,
            y=y0,
            x=x0,
            bucket_count=bucket_count,
            use_buckets=use_buckets,
            use_s_sets=use_s_sets,
            s_sets=s_sets,
            s_clip=s_clip,
            o_min=o_min,
            o_max=o_max,
            quantized=quantized,
            topk=topk,
            lsh_budget=lsh_budget,
            regression=regression,
        )
        leaf_yx_list.append((y0, x0))
        leaf_pool_list.append(pi)
        leaf_dom_list.append(dom)
        leaf_tf_list.append(tf)
        if quantized:
            leaf_codes_q_list.append(codes.astype(np.uint8, copy=False))
        else:
            leaf_codes_f_list.append(codes.astype(np.float32, copy=False))
        return mse

    def encode_node(y0: int, x0: int, block0: int) -> None:
        mse = emit_leaf(y0, x0, block0)
        if mse <= max_mse or block0 <= min_block:
            return

        leaf_yx_list.pop()
        leaf_pool_list.pop()
        leaf_dom_list.pop()
        leaf_tf_list.pop()
        if quantized:
            leaf_codes_q_list.pop()
        else:
            leaf_codes_f_list.pop()

        nb = block0 // 2
        encode_node(y0, x0, nb)
        encode_node(y0, x0 + nb, nb)
        encode_node(y0 + nb, x0, nb)
        encode_node(y0 + nb, x0 + nb, nb)

    if use_quadtree:
        for y in range(0, h, max_block):
            for x in range(0, w, max_block):
                encode_node(y, x, max_block)
    else:
        b0 = max_block
        for y in range(0, h, b0):
            for x in range(0, w, b0):
                emit_leaf(y, x, b0)

    leaf_yx = np.asarray(leaf_yx_list, dtype=np.uint16)
    leaf_pool = np.asarray(leaf_pool_list, dtype=np.uint8)
    leaf_dom = np.asarray(leaf_dom_list, dtype=np.uint32)
    leaf_tf = np.asarray(leaf_tf_list, dtype=np.uint8)

    pool_blocks = np.asarray([p.block for p in pools], dtype=np.uint16)
    pool_strides = np.asarray([p.stride for p in pools], dtype=np.uint16)
    pool_offsets = np.zeros((len(pools) + 1,), dtype=np.uint32)
    for i, p in enumerate(pools):
        pool_offsets[i + 1] = pool_offsets[i] + np.uint32(p.domain_yx.shape[0])
    domain_yx = np.concatenate([p.domain_yx for p in pools], axis=0).astype(
        np.uint16, copy=False
    )

    leaf_codes_q: NDArray[np.uint8] | None = None
    leaf_codes_f: NDArray[np.float32] | None = None
    if quantized:
        leaf_codes_q = np.stack(leaf_codes_q_list, axis=0).astype(np.uint8, copy=False)
    else:
        leaf_codes_f = np.stack(leaf_codes_f_list, axis=0).astype(
            np.float32, copy=False
        )

    return FractalCode(
        height=h,
        width=w,
        orig_height=orig_h,
        orig_width=orig_w,
        channels=c,
        pool_blocks=pool_blocks,
        pool_strides=pool_strides,
        pool_offsets=pool_offsets,
        domain_yx=domain_yx,
        leaf_yx=leaf_yx,
        leaf_pool=leaf_pool,
        leaf_dom=leaf_dom,
        leaf_tf=leaf_tf,
        quantized=bool(quantized),
        s_clip=float(s_clip),
        o_min=float(o_min),
        o_max=float(o_max),
        leaf_codes_q=leaf_codes_q,
        leaf_codes_f=leaf_codes_f,
        iterations_hint=int(iterations_hint),
    )


def encode_to_file(
    input_path: Path,
    output_path: Path,
    min_block: int = 4,
    max_block: int = 16,
    stride: int = 4,
    use_quadtree: bool = False,
    max_mse: float = 0.0025,
    use_buckets: bool = False,
    bucket_count: int = 8,
    use_s_sets: bool = False,
    topk: int = 64,
    backend: str = "dot",
    lsh_budget: int = 2048,
    entropy_thresh: float = 0.0,
    quantized: bool = False,
    s_clip: float = 0.99,
    o_min: float = -0.5,
    o_max: float = 1.5,
    pca_dim: int = 16,
    lsh_planes: int = 16,
    seed: int = 0,
    max_domains: int | None = None,
    block: int | None = None,
    transform_ids: tuple[int, ...] | None = None,
    regression: RegressionKind | str = "linear",
) -> None:
    img = load_image(input_path)
    code = encode_array(
        img=img,
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
        transform_ids=transform_ids,
        block=block,
        regression=regression,
    )
    save_code(output_path, code)
