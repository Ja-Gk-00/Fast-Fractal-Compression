from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from fastfractal import _cext  # type: ignore
from fastfractal.core.blocks import extract_range, iter_domains
from fastfractal.core.search import (
    LSHIndex,
    SearchBackend,
    fit_pca,
    normalize_rows,
    topk_from_subset,
)
from fastfractal.core.transforms import apply_transform_2d
from fastfractal.core.types import FractalCode
from fastfractal.io.codebook import save_code
from fastfractal.io.imageio import load_image
from fastfractal.utils.entropy import entropy01


def _has_cext() -> bool:
    return hasattr(_cext, "downsample2x2") and hasattr(_cext, "linreg_error")


def downsample2x2(x: NDArray[np.float32]) -> NDArray[np.float32]:
    if _has_cext():
        return _cext.downsample2x2(x)  # type: ignore[no-any-return]
    h, w = x.shape
    return (
        x.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3)).astype(np.float32, copy=False)
    )


def linreg_error(
    d: NDArray[np.float32], r: NDArray[np.float32]
) -> tuple[float, float, float]:
    if _has_cext():
        s, o, e = _cext.linreg_error(d, r)  # type: ignore[misc]
        return float(s), float(o), float(e)
    dv = d.astype(np.float64, copy=False)
    rv = r.astype(np.float64, copy=False)
    n = float(dv.size)
    sumD = float(dv.sum())
    sumR = float(rv.sum())
    sumDD = float((dv * dv).sum())
    sumRR = float((rv * rv).sum())
    sumRD = float((dv * rv).sum())
    denom = n * sumDD - sumD * sumD
    if abs(denom) < 1e-18:
        s = 0.0
        o = sumR / n
    else:
        s = (n * sumRD - sumD * sumR) / denom
        o = (sumR - s * sumD) / n
    err = (
        sumRR
        + s * s * sumDD
        + n * o * o
        - 2.0 * s * sumRD
        - 2.0 * o * sumR
        + 2.0 * s * o * sumD
    )
    return float(s), float(o), float(err)


def rgb_to_luma(img: NDArray[np.float32]) -> NDArray[np.float32]:
    if img.ndim == 2:
        return img
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    return (0.2989 * r + 0.5870 * g + 0.1140 * b).astype(np.float32, copy=False)


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


def var01(x: NDArray[np.float32]) -> float:
    m = float(np.mean(x))
    d = x.astype(np.float32, copy=False) - np.float32(m)
    return float(np.mean(d * d))


def bucket_id(ent: float, var: float, bucket_count: int) -> int:
    vn = float(np.clip(var / 0.25, 0.0, 1.0))
    s = 0.5 * float(np.clip(ent, 0.0, 1.0)) + 0.5 * vn
    i = int(s * bucket_count)
    if i >= bucket_count:
        return bucket_count - 1
    if i < 0:
        return 0
    return i


def default_s_sets(bucket_count: int, s_clip: float) -> list[list[float]]:
    out: list[list[float]] = []
    for i in range(bucket_count):
        t = 0.0 if bucket_count <= 1 else float(i) / float(bucket_count - 1)
        mx = 0.2 + 0.79 * t
        mx = float(np.clip(mx, 0.05, s_clip))
        vals = np.linspace(0.0, mx, 5, dtype=np.float64)
        sset = [float(v) for v in vals]
        if i >= bucket_count // 2:
            neg = [-float(v) for v in vals[1:3]]
            sset = neg + sset
        out.append(sset)
    return out


def quant_s(s: float, s_clip: float) -> int:
    sc = float(np.clip(s, -s_clip, s_clip))
    q = int(np.rint((sc + s_clip) * 255.0 / (2.0 * s_clip)))
    if q < 0:
        return 0
    if q > 255:
        return 255
    return q


def dequant_s(q: int, s_clip: float) -> float:
    return float(q) * (2.0 * s_clip) / 255.0 - s_clip


def quant_o(o: float, o_min: float, o_max: float) -> int:
    oc = float(np.clip(o, o_min, o_max))
    q = int(np.rint((oc - o_min) * 255.0 / (o_max - o_min)))
    if q < 0:
        return 0
    if q > 255:
        return 255
    return q


def dequant_o(q: int, o_min: float, o_max: float) -> float:
    return o_min + float(q) * (o_max - o_min) / 255.0


def choose_s_from_set(s: float, sset: list[float]) -> float:
    if not sset:
        return s
    arr = np.asarray(sset, dtype=np.float64)
    i = int(np.argmin(np.abs(arr - float(s))))
    return float(arr[i])


@dataclass(frozen=True, slots=True)
class PoolRuntime:
    block: int
    stride: int
    domain_yx: NDArray[np.uint16]
    tf_flat: NDArray[np.float32]
    proxy_mat: NDArray[np.float32]
    map_dom: NDArray[np.uint32]
    map_tf: NDArray[np.uint8]
    entry_bucket: NDArray[np.uint8]
    bucket_entries: list[NDArray[np.int64]]
    backend: str
    lsh: LSHIndex | None
    pca_mean: NDArray[np.float32] | None
    pca_basis: NDArray[np.float32] | None


def build_pool(
    img: NDArray[np.float32],
    block: int,
    stride: int,
    entropy_thresh: float,
    bucket_count: int,
    use_buckets: bool,
    backend: str,
    pca_dim: int,
    lsh_planes: int,
    seed: int,
    max_domains: int | None,
) -> PoolRuntime:
    h = int(img.shape[0])
    w = int(img.shape[1])
    c = 1 if img.ndim == 2 else int(img.shape[2])

    luma = rgb_to_luma(img)
    dom_xy: list[tuple[int, int]] = []
    dom_proxy: list[NDArray[np.float32]] = []
    dom_ch: list[NDArray[np.float32]] = []
    dom_ent: list[float] = []
    dom_var: list[float] = []

    for _, y, x in iter_domains(h, w, block, stride):
        d2 = luma[y : y + 2 * block, x : x + 2 * block]
        ds = downsample2x2(d2)
        if entropy_thresh > 0.0:
            if entropy01(ds) < entropy_thresh:
                continue

        dom_xy.append((y, x))
        dom_proxy.append(ds.astype(np.float32, copy=False))

        if use_buckets:
            dom_ent.append(entropy01(ds))
            dom_var.append(var01(ds))

        if c == 1:
            dch = downsample2x2(img[y : y + 2 * block, x : x + 2 * block]).astype(
                np.float32, copy=False
            )
            dom_ch.append(dch[None, :, :])
        else:
            chans: list[NDArray[np.float32]] = []
            for ch in range(c):
                dcc = img[y : y + 2 * block, x : x + 2 * block, ch]
                chans.append(downsample2x2(dcc).astype(np.float32, copy=False))
            dom_ch.append(np.stack(chans, axis=0))

    if max_domains is not None and len(dom_xy) > max_domains:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(dom_xy), size=max_domains, replace=False)
        keep.sort()
        dom_xy = [dom_xy[i] for i in keep]
        dom_proxy = [dom_proxy[i] for i in keep]
        dom_ch = [dom_ch[i] for i in keep]
        if use_buckets:
            dom_ent = [dom_ent[i] for i in keep]
            dom_var = [dom_var[i] for i in keep]

    dcount = len(dom_xy)
    if dcount == 0:
        raise ValueError("domain pool empty")

    yx = np.asarray(dom_xy, dtype=np.uint16)
    n_pix = block * block
    tf_flat = np.empty((dcount * 8, c, n_pix), dtype=np.float32)
    proxy_raw = np.empty((dcount * 8, n_pix), dtype=np.float32)
    map_dom = np.empty((dcount * 8,), dtype=np.uint32)
    map_tf = np.empty((dcount * 8,), dtype=np.uint8)

    entry_bucket = np.zeros((dcount * 8,), dtype=np.uint8)
    if use_buckets:
        for di in range(dcount):
            bid = bucket_id(dom_ent[di], dom_var[di], bucket_count)
            for t in range(8):
                entry_bucket[di * 8 + t] = np.uint8(bid)

    for di in range(dcount):
        domc = dom_ch[di]
        dproxy = dom_proxy[di]
        for t in range(8):
            k = di * 8 + t
            map_dom[k] = np.uint32(di)
            map_tf[k] = np.uint8(t)
            proxy_raw[k, :] = (
                apply_transform_2d(dproxy, t).reshape(-1).astype(np.float32, copy=False)
            )
            for ch in range(c):
                tf_flat[k, ch, :] = (
                    apply_transform_2d(domc[ch], t)
                    .reshape(-1)
                    .astype(np.float32, copy=False)
                )

    proxy_mat = normalize_rows(proxy_raw.astype(np.float32, copy=False))

    bucket_entries: list[NDArray[np.int64]] = []
    if use_buckets:
        for b in range(bucket_count):
            idx = np.nonzero(entry_bucket == np.uint8(b))[0].astype(
                np.int64, copy=False
            )
            bucket_entries.append(idx)
    else:
        bucket_entries = [np.arange(proxy_mat.shape[0], dtype=np.int64)]

    if backend not in SearchBackend:
        raise ValueError("bad backend")

    lsh: LSHIndex | None = None
    pca_mean: NDArray[np.float32] | None = None
    pca_basis: NDArray[np.float32] | None = None

    if backend == "lsh":
        lsh = LSHIndex.build(proxy_mat, planes=lsh_planes, seed=seed)
    elif backend == "pca_lsh":
        p = fit_pca(
            proxy_mat, dim=pca_dim, sample=min(5000, proxy_mat.shape[0]), seed=seed
        )
        pca_mean = p.mean
        pca_basis = p.basis
        proj = p.project_matrix(proxy_mat)
        lsh = LSHIndex.build(proj, planes=lsh_planes, seed=seed)

    return PoolRuntime(
        block=block,
        stride=stride,
        domain_yx=yx,
        tf_flat=tf_flat,
        proxy_mat=proxy_mat,
        map_dom=map_dom,
        map_tf=map_tf,
        entry_bucket=entry_bucket,
        bucket_entries=bucket_entries,
        backend=backend,
        lsh=lsh,
        pca_mean=pca_mean,
        pca_basis=pca_basis,
    )


def pool_query_candidates(
    pool: PoolRuntime,
    q: NDArray[np.float32],
    bid: int,
    topk: int,
    lsh_budget: int,
) -> NDArray[np.int64]:
    if pool.backend == "dot":
        subset = pool.bucket_entries[bid]
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
) -> tuple[int, int, NDArray[np.uint8] | NDArray[np.float32], float]:
    b = pool.block
    c = 1 if img.ndim == 2 else int(img.shape[2])

    if use_buckets:
        rproxy2 = extract_range(luma, y, x, b).astype(np.float32, copy=False)
        ent = entropy01(rproxy2)
        vr = var01(rproxy2)
        bid = bucket_id(ent, vr, bucket_count)
    else:
        bid = 0

    rproxy = extract_range(luma, y, x, b).reshape(1, -1).astype(np.float32, copy=False)
    q = normalize_rows(rproxy)[0]
    cand = pool_query_candidates(pool, q, bid, topk=topk, lsh_budget=lsh_budget)

    n_pix = b * b
    best_mse = float("inf")
    best_dom = 0
    best_tf = 0

    if quantized:
        best_q = np.zeros((c, 2), dtype=np.uint8)
    else:
        best_f = np.zeros((c, 2), dtype=np.float32)

    def post_s(s0: float) -> float:
        if use_s_sets and use_buckets:
            return float(np.clip(choose_s_from_set(s0, s_sets[bid]), -s_clip, s_clip))
        return float(np.clip(s0, -s_clip, s_clip))

    if c == 1:
        r = extract_range(img, y, x, b).reshape(-1).astype(np.float32, copy=False)
        for ci in cand:
            domv = pool.tf_flat[int(ci), 0, :]
            s0, o0, _ = linreg_error(domv, r)
            s1 = post_s(s0)
            o1 = float(np.clip(o0, o_min, o_max))
            if quantized:
                qs = quant_s(s1, s_clip)
                qo = quant_o(o1, o_min, o_max)
                s2 = dequant_s(qs, s_clip)
                o2 = dequant_o(qo, o_min, o_max)
                diff = (np.float32(s2) * domv + np.float32(o2) - r).astype(
                    np.float32, copy=False
                )
                mse = float(np.dot(diff, diff) / float(n_pix))
                if mse < best_mse:
                    best_mse = mse
                    best_dom = int(pool.map_dom[int(ci)])
                    best_tf = int(pool.map_tf[int(ci)])
                    best_q[0, 0] = np.uint8(qs)
                    best_q[0, 1] = np.uint8(qo)
            else:
                diff = (np.float32(s1) * domv + np.float32(o1) - r).astype(
                    np.float32, copy=False
                )
                mse = float(np.dot(diff, diff) / float(n_pix))
                if mse < best_mse:
                    best_mse = mse
                    best_dom = int(pool.map_dom[int(ci)])
                    best_tf = int(pool.map_tf[int(ci)])
                    best_f[0, 0] = np.float32(s1)
                    best_f[0, 1] = np.float32(o1)
    else:
        rblk = img[y : y + b, x : x + b, :].astype(np.float32, copy=False)
        rflat = np.transpose(rblk, (2, 0, 1)).reshape(c, -1)
        for ci in cand:
            mse_sum = 0.0
            if quantized:
                qtmp = np.zeros((c, 2), dtype=np.uint8)
            else:
                ftmp = np.zeros((c, 2), dtype=np.float32)
            for ch in range(c):
                domv = pool.tf_flat[int(ci), ch, :]
                s0, o0, _ = linreg_error(domv, rflat[ch])
                s1 = post_s(s0)
                o1 = float(np.clip(o0, o_min, o_max))
                if quantized:
                    qs = quant_s(s1, s_clip)
                    qo = quant_o(o1, o_min, o_max)
                    s2 = dequant_s(qs, s_clip)
                    o2 = dequant_o(qo, o_min, o_max)
                    diff = (np.float32(s2) * domv + np.float32(o2) - rflat[ch]).astype(
                        np.float32, copy=False
                    )
                    mse_sum += float(np.dot(diff, diff) / float(n_pix))
                    qtmp[ch, 0] = np.uint8(qs)
                    qtmp[ch, 1] = np.uint8(qo)
                else:
                    diff = (np.float32(s1) * domv + np.float32(o1) - rflat[ch]).astype(
                        np.float32, copy=False
                    )
                    mse_sum += float(np.dot(diff, diff) / float(n_pix))
                    ftmp[ch, 0] = np.float32(s1)
                    ftmp[ch, 1] = np.float32(o1)

            mse = mse_sum / float(c)
            if mse < best_mse:
                best_mse = mse
                best_dom = int(pool.map_dom[int(ci)])
                best_tf = int(pool.map_tf[int(ci)])
                if quantized:
                    best_q = qtmp
                else:
                    best_f = ftmp

    if quantized:
        return best_dom, best_tf, best_q, best_mse
    return best_dom, best_tf, best_f, best_mse


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
    b = max_block
    while b >= min_block:
        blocks.append(b)
        if b == min_block:
            break
        b //= 2

    pools: list[PoolRuntime] = []
    for b in blocks:
        pools.append(
            build_pool(
                img=img2,
                block=b,
                stride=stride,
                entropy_thresh=entropy_thresh,
                bucket_count=bucket_count,
                use_buckets=use_buckets,
                backend=backend,
                pca_dim=pca_dim,
                lsh_planes=lsh_planes,
                seed=seed + b,
                max_domains=max_domains,
            )
        )

    if len(pools) > 255:
        raise ValueError("too many pools")

    leaf_yx_list: list[tuple[int, int]] = []
    leaf_pool_list: list[int] = []
    leaf_dom_list: list[int] = []
    leaf_tf_list: list[int] = []
    leaf_codes_q_list: list[NDArray[np.uint8]] = []
    leaf_codes_f_list: list[NDArray[np.float32]] = []

    def pool_index_for_block(block: int) -> int:
        for i, p in enumerate(pools):
            if p.block == block:
                return i
        raise ValueError("missing pool")

    def emit_leaf(y0: int, x0: int, block0: int) -> float:
        pi = pool_index_for_block(block0)
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
    )
    save_code(output_path, code)


def encode(
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
) -> None:
    encode_to_file(
        input_path=input_path,
        output_path=output_path,
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
