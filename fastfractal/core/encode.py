from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

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
from fastfractal.core.types import FractalCode, PoolRuntime
from fastfractal.io.codebook import save_code
from fastfractal.io.imageio import load_image
from fastfractal.utils.entropy import entropy01

if TYPE_CHECKING:
    from numpy.typing import NDArray


_CANONICAL_TRANSFORMS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7)

_HAS_LINREG = hasattr(_cext, "linreg_error")
_HAS_DOWNSAMPLE = hasattr(_cext, "downsample2x2")
_HAS_TOPK = hasattr(_cext, "topk_from_subset")
_HAS_ENCODE_LEAF = hasattr(_cext, "encode_leaf_best")


def _clipf(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _normalize_transform_ids(
    transform_ids: tuple[int, ...] | None | str,
) -> tuple[int, ...]:
    if transform_ids is None:
        return _CANONICAL_TRANSFORMS
    if transform_ids == "all":
        return tuple(range(0, 15))

    ids: list[int] = []
    seen: set[int] = set()
    for t in transform_ids:
        ti = int(t)
        if ti < 0 or ti > 14:
            raise ValueError(
                f"transform_ids: invalid transform id {ti}; expected 0..14"
            )
        if ti not in seen:
            ids.append(ti)
            seen.add(ti)

    if not ids:
        raise ValueError("transform_ids: must contain at least one transform id")
    return tuple(ids)


def _downsample2x2_f32(x: NDArray[np.float32]) -> NDArray[np.float32]:
    if _HAS_DOWNSAMPLE:
        return _cext.downsample2x2(x)  # type: ignore[attr-defined, unused-ignore, no-any-return]
    y = (x[0::2, 0::2] + x[1::2, 0::2] + x[0::2, 1::2] + x[1::2, 1::2]) * np.float32(
        0.25
    )
    return y.astype(np.float32, copy=False)


def linreg_error(
    d: NDArray[np.float32], r: NDArray[np.float32]
) -> tuple[float, float, float]:
    if _HAS_LINREG:
        s, o, e = _cext.linreg_error(d, r)
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
    return np.asarray(0.2989 * r + 0.5870 * g + 0.1140 * b, dtype=np.float32)


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
    xf = x.ravel()
    n = int(xf.size)
    if n <= 0:
        return 0.0
    s1 = float(xf.sum(dtype=np.float64))
    s2 = float(np.dot(xf, xf))
    m = s1 / float(n)
    v = (s2 / float(n)) - (m * m)
    return 0.0 if v < 0.0 else float(v)


def bucket_id(ent: float, var: float, bucket_count: int) -> int:
    vn = _clip01(var / 0.25)
    s = 0.5 * _clip01(ent) + 0.5 * vn
    i = int(s * float(bucket_count))
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


def quant_s(s: float, s_clip: float) -> int:
    sc = _clipf(s, -float(s_clip), float(s_clip))
    q = int(round((sc + float(s_clip)) * 255.0 / (2.0 * float(s_clip))))
    if q < 0:
        return 0
    if q > 255:
        return 255
    return q


def dequant_s(q: int, s_clip: float) -> float:
    return float(q) * (2.0 * float(s_clip)) / 255.0 - float(s_clip)


def quant_o(o: float, o_min: float, o_max: float) -> int:
    oc = _clipf(o, float(o_min), float(o_max))
    q = int(round((oc - float(o_min)) * 255.0 / (float(o_max) - float(o_min))))
    if q < 0:
        return 0
    if q > 255:
        return 255
    return q


def dequant_o(q: int, o_min: float, o_max: float) -> float:
    return float(o_min) + float(q) * (float(o_max) - float(o_min)) / 255.0


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
    transform_ids: tuple[int, ...] | None,
    *,
    luma: NDArray[np.float32] | None = None,
    precompute_stats: bool = True,
) -> PoolRuntime:
    h = int(img.shape[0])
    w = int(img.shape[1])
    c = 1 if img.ndim == 2 else int(img.shape[2])

    if luma is None:
        luma = rgb_to_luma(img)

    use_ds_slices = (int(stride) & 1) == 0

    if use_ds_slices:
        stride2 = int(stride) // 2

        luma_ds = _downsample2x2_f32(luma)

        if c == 1:
            img_ds_2d = _downsample2x2_f32(img)  # (H/2, W/2)
            img_ds_3d = None
        else:
            oh, ow = int(luma_ds.shape[0]), int(luma_ds.shape[1])
            img_ds_3d = np.empty((c, oh, ow), dtype=np.float32)
            for ch in range(c):
                img_ds_3d[ch, :, :] = _downsample2x2_f32(img[:, :, ch])
            img_ds_2d = None

        oh = int(luma_ds.shape[0])
        ow = int(luma_ds.shape[1])
        ny = (oh - int(block)) // stride2 + 1
        nx = (ow - int(block)) // stride2 + 1
        if ny <= 0 or nx <= 0:
            raise ValueError("domain pool empty")
        nmax = int(ny * nx)

        yx_buf = np.empty((nmax, 2), dtype=np.uint16)
        proxy_buf = np.empty((nmax, block, block), dtype=np.float32)
        domc_buf = np.empty((nmax, c, block, block), dtype=np.float32)

        need_entropy = (entropy_thresh > 0.0) or use_buckets
        ent_buf = np.empty((nmax,), dtype=np.float32) if need_entropy else None
        var_buf = np.empty((nmax,), dtype=np.float32) if use_buckets else None

        count = 0
        for y2 in range(0, oh - block + 1, stride2):
            y0 = int(y2) * 2
            for x2 in range(0, ow - block + 1, stride2):
                x0 = int(x2) * 2

                ds = luma_ds[y2 : y2 + block, x2 : x2 + block]

                ent = 0.0
                if need_entropy:
                    ent = float(entropy01(ds))
                    if entropy_thresh > 0.0 and ent < float(entropy_thresh):
                        continue

                yx_buf[count, 0] = np.uint16(y0)
                yx_buf[count, 1] = np.uint16(x0)
                proxy_buf[count, :, :] = ds

                if ent_buf is not None:
                    ent_buf[count] = np.float32(ent)
                if var_buf is not None:
                    var_buf[count] = np.float32(var01(ds))

                if c == 1:
                    domc_buf[count, 0, :, :] = img_ds_2d[
                        y2 : y2 + block, x2 : x2 + block
                    ]  # type: ignore[index]
                else:
                    for ch in range(c):
                        domc_buf[count, ch, :, :] = img_ds_3d[
                            ch, y2 : y2 + block, x2 : x2 + block
                        ]  # type: ignore[index]

                count += 1

        dcount = int(count)
        if dcount == 0:
            raise ValueError("domain pool empty")

        yx = yx_buf[:dcount, :]
        dom_proxy = proxy_buf[:dcount, :, :]
        dom_ch = domc_buf[:dcount, :, :, :]

    else:
        ny = (h - 2 * block) // stride + 1
        nx = (w - 2 * block) // stride + 1
        if ny <= 0 or nx <= 0:
            raise ValueError("domain pool empty")
        nmax = int(ny * nx)

        yx_buf = np.empty((nmax, 2), dtype=np.uint16)
        proxy_buf = np.empty((nmax, block, block), dtype=np.float32)
        domc_buf = np.empty((nmax, c, block, block), dtype=np.float32)

        need_entropy = (entropy_thresh > 0.0) or use_buckets
        ent_buf = np.empty((nmax,), dtype=np.float32) if need_entropy else None
        var_buf = np.empty((nmax,), dtype=np.float32) if use_buckets else None

        count = 0
        for _, y, x in iter_domains(h, w, block, stride):
            ds = _downsample2x2_f32(luma[y : y + 2 * block, x : x + 2 * block]).astype(
                np.float32, copy=False
            )

            ent = 0.0
            if need_entropy:
                ent = float(entropy01(ds))
                if entropy_thresh > 0.0 and ent < float(entropy_thresh):
                    continue

            yx_buf[count, 0] = np.uint16(y)
            yx_buf[count, 1] = np.uint16(x)
            proxy_buf[count, :, :] = ds

            if ent_buf is not None:
                ent_buf[count] = np.float32(ent)
            if var_buf is not None:
                var_buf[count] = np.float32(var01(ds))

            if c == 1:
                dch = _downsample2x2_f32(
                    img[y : y + 2 * block, x : x + 2 * block]
                ).astype(np.float32, copy=False)
                domc_buf[count, 0, :, :] = dch
            else:
                for ch in range(c):
                    dcc = img[y : y + 2 * block, x : x + 2 * block, ch]
                    domc_buf[count, ch, :, :] = _downsample2x2_f32(dcc).astype(
                        np.float32, copy=False
                    )

            count += 1

        dcount = int(count)
        if dcount == 0:
            raise ValueError("domain pool empty")

        yx = yx_buf[:dcount, :]
        dom_proxy = proxy_buf[:dcount, :, :]
        dom_ch = domc_buf[:dcount, :, :, :]

    if max_domains is not None and int(yx.shape[0]) > int(max_domains):
        rng = np.random.default_rng(seed)
        keep = rng.choice(int(yx.shape[0]), size=int(max_domains), replace=False)
        keep.sort()
        yx = yx[keep]
        dom_proxy = dom_proxy[keep]
        dom_ch = dom_ch[keep]
        if ent_buf is not None:
            ent_buf = ent_buf[: int(ent_buf.shape[0])][keep]
        if var_buf is not None:
            var_buf = var_buf[: int(var_buf.shape[0])][keep]
        dcount = int(yx.shape[0])
    else:
        dcount = int(yx.shape[0])

    tids = _normalize_transform_ids(transform_ids)
    n_tf = int(len(tids))
    n_pix = int(block * block)

    n_entries = int(dcount * n_tf)
    tf_flat = np.empty((n_entries, c, n_pix), dtype=np.float32)
    proxy_raw = np.empty((n_entries, n_pix), dtype=np.float32)

    map_dom = np.repeat(np.arange(dcount, dtype=np.uint32), n_tf)
    map_tf = np.tile(np.asarray(tids, dtype=np.uint8), dcount)

    if use_buckets:
        if ent_buf is None or var_buf is None:
            raise RuntimeError("internal: missing bucket stats buffers")
        bids = np.empty((dcount,), dtype=np.uint8)
        for di in range(dcount):
            bids[di] = np.uint8(
                bucket_id(float(ent_buf[di]), float(var_buf[di]), int(bucket_count))
            )
        entry_bucket = np.repeat(bids, n_tf)
    else:
        entry_bucket = np.zeros((n_entries,), dtype=np.uint8)

    for di in range(dcount):
        dproxy = dom_proxy[di]
        domc = dom_ch[di]
        base = di * n_tf
        for ti, t in enumerate(tids):
            k = base + ti
            proxy_raw[k, :].reshape(block, block)[:] = apply_transform_2d(dproxy, t)
            for ch in range(c):
                tf_flat[k, ch, :].reshape(block, block)[:] = apply_transform_2d(
                    domc[ch], t
                )

    proxy_mat = normalize_rows(proxy_raw)

    bucket_entries: list[NDArray[np.int32]] = []
    if use_buckets:
        for b in range(int(bucket_count)):
            idx = np.nonzero(entry_bucket == np.uint8(b))[0].astype(
                np.int32, copy=False
            )
            bucket_entries.append(idx)
    else:
        bucket_entries = [np.arange(proxy_mat.shape[0], dtype=np.int32)]

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

    tf_sum: NDArray[np.float64] | None = None
    tf_sum2: NDArray[np.float64] | None = None
    if precompute_stats:
        tf_sum = tf_flat.sum(axis=2, dtype=np.float64)
        tf_sum2 = np.einsum("kcn,kcn->kc", tf_flat, tf_flat, dtype=np.float64)

    return PoolRuntime(
        block=block,
        stride=stride,
        domain_yx=yx,
        tf_flat=tf_flat,
        tf_sum=tf_sum,
        tf_sum2=tf_sum2,
        proxy_mat=proxy_mat,
        map_dom=map_dom,
        map_tf=map_tf,
        entry_bucket=entry_bucket,
        bucket_entries=bucket_entries,  # type: ignore[arg-type]
        backend=backend,
        lsh=lsh,
        pca_mean=pca_mean,
        pca_basis=pca_basis,
        transform_ids=tids,
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

            s0, o0, _ = linreg_error(domv, r)
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
    )
    save_code(output_path, code)
