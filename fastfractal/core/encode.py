from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from fastfractal._cext_backend import get, has
from fastfractal.core.blocks import domains_yx
from fastfractal.core.transforms import apply_transform_2d
from fastfractal.core.types import FractalCode
from fastfractal.io.codebook import save_code
from fastfractal.io.imageio import load_image

import os
from fastfractal._cext_backend import cext


def downsample2x2(dom: np.ndarray) -> np.ndarray:
    if cext.has("downsample2x2"):
        return np.asarray(cext.call("downsample2x2", dom), dtype=np.float32)

    dom = np.asarray(dom, dtype=np.float32)
    h, w = dom.shape[:2]
    if h % 2 or w % 2:
        raise ValueError("downsample2x2 expects even dimensions")
    return 0.25 * (
        dom[0::2, 0::2] + dom[1::2, 0::2] + dom[0::2, 1::2] + dom[1::2, 1::2]
    )


def linreg_error(d: np.ndarray, r: np.ndarray) -> Tuple[float, float, float]:
    """
    Fit r ≈ s*d + o. Returns (s, o, mse).
    """
    if cext.has("linreg_error"):
        s, o, err = cext.call("linreg_error", d, r)
        return float(s), float(o), float(err)

    d = np.asarray(d, dtype=np.float32).ravel()
    r = np.asarray(r, dtype=np.float32).ravel()
    if d.size != r.size:
        raise ValueError("linreg_error: d and r must have same number of elements")
    n = float(d.size)
    sd = float(d.sum())
    sr = float(r.sum())
    sdd = float((d * d).sum())
    sdr = float((d * r).sum())

    denom = n * sdd - sd * sd
    if denom == 0.0:
        s = 0.0
        o = sr / n
    else:
        s = (n * sdr - sd * sr) / denom
        o = (sr - s * sd) / n

    diff = s * d + o - r
    mse = float((diff * diff).mean())
    return float(s), float(o), mse


def _has_topk_dot() -> bool:
    return cext.has("topk_dot")


def _topk_dot_numpy(d: np.ndarray, q: np.ndarray, k: int) -> np.ndarray:
    scores = d @ q
    if k <= 0:
        return np.empty((0,), dtype=np.int64)
    if k >= scores.size:
        return np.argsort(scores)[::-1].astype(np.int64, copy=False)
    idx = np.argpartition(scores, -k)[-k:]
    idx = idx[np.argsort(scores[idx])[::-1]]
    return idx.astype(np.int64, copy=False)


def _cext_topk_dot(
    ranges: NDArray[np.float32],
    domains: NDArray[np.float32],
    topk: int,
) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
    r = np.ascontiguousarray(ranges, dtype=np.float32)
    d = np.ascontiguousarray(domains, dtype=np.float32)

    if r.ndim == 1:
        r = r.reshape(1, -1)

    try:
        out = cext.call("topk_dot", r, d, int(topk))
        idx, val = out
        idx = np.asarray(idx, dtype=np.int64)
        val = np.asarray(val, dtype=np.float32)
        if idx.ndim == 1:
            idx = idx.reshape(1, -1)
            val = val.reshape(1, -1)
        return idx, val
    except (TypeError, ValueError):
        pass

    if r.shape[0] == 1:
        q = np.ascontiguousarray(r[0], dtype=np.float32).ravel()
        out = cext.call("topk_dot", d, q, int(topk))

        if isinstance(out, tuple) and len(out) == 2:
            idx, val = out
            idx = np.asarray(idx, dtype=np.int64).ravel()
            val = np.asarray(val, dtype=np.float32).ravel()
        else:
            idx = np.asarray(out, dtype=np.int64).ravel()

            val = (d[idx] @ q).astype(np.float32, copy=False)

        return idx.reshape(1, -1), val.reshape(1, -1)

    scores = r @ d.T
    K = scores.shape[1]
    k = int(topk)
    if k >= K:
        idx = np.argsort(-scores, axis=1)
        val = np.take_along_axis(scores, idx, axis=1).astype(np.float32, copy=False)
        return idx.astype(np.int64, copy=False), val

    part = np.argpartition(scores, K - k, axis=1)[:, -k:]
    row = np.arange(scores.shape[0])[:, None]
    vals = scores[row, part]
    order = np.argsort(-vals, axis=1)
    idx = part[row, order]
    val = vals[row, order].astype(np.float32, copy=False)
    return idx.astype(np.int64, copy=False), val


def _pad_to_multiple(
    img: NDArray[np.float32], block: int
) -> tuple[NDArray[np.float32], int, int]:
    """
    Pads bottom/right with edge values to make H,W multiples of block.
    Returns (padded_img, orig_h, orig_w).
    """
    h = int(img.shape[0])
    w = int(img.shape[1])
    ph = (block - (h % block)) % block
    pw = (block - (w % block)) % block
    if ph == 0 and pw == 0:
        return img.astype(np.float32, copy=False), h, w

    if img.ndim == 2:
        out = np.pad(img, ((0, ph), (0, pw)), mode="edge").astype(
            np.float32, copy=False
        )
    else:
        out = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="edge").astype(
            np.float32, copy=False
        )
    return out, h, w


def _to_gray(x: NDArray[np.float32]) -> NDArray[np.float32]:
    if x.ndim == 2:
        return x
    r = x[:, :, 0]
    g = x[:, :, 1]
    b = x[:, :, 2]
    return (0.2989 * r + 0.5870 * g + 0.1140 * b).astype(np.float32, copy=False)


def _normalize_rows(mat: NDArray[np.float32], eps: float = 1e-8) -> NDArray[np.float32]:
    """
    Row-wise zero-mean + L2 normalize. Gives cosine-sim via dot product.
    """
    m = mat.astype(np.float32, copy=False)
    m = m - m.mean(axis=1, keepdims=True)
    n = np.sqrt(np.sum(m * m, axis=1, keepdims=True))
    n = np.maximum(n, eps)
    return (m / n).astype(np.float32, copy=False)


def _normalize_vec(v: NDArray[np.float32], eps: float = 1e-8) -> NDArray[np.float32]:
    vv = v.astype(np.float32, copy=False).reshape(-1)
    vv = vv - np.float32(vv.mean())
    n = float(np.sqrt(np.dot(vv, vv)))
    if n < eps:
        return vv * 0.0
    return (vv / np.float32(n)).astype(np.float32, copy=False)


def _quant_s(s: float, s_clip: float) -> int:
    sc = float(np.clip(s, -s_clip, s_clip))
    q = int(np.rint((sc + s_clip) * 255.0 / (2.0 * s_clip)))
    return int(np.clip(q, 0, 255))


def _dequant_s(q: int, s_clip: float) -> float:
    return float(q) * (2.0 * s_clip) / 255.0 - s_clip


def _quant_o(o: float, o_min: float, o_max: float) -> int:
    oc = float(np.clip(o, o_min, o_max))
    q = int(np.rint((oc - o_min) * 255.0 / (o_max - o_min)))
    return int(np.clip(q, 0, 255))


def _dequant_o(q: int, o_min: float, o_max: float) -> float:
    return o_min + float(q) * (o_max - o_min) / 255.0


@dataclass(frozen=True)
class _Pool:
    block: int
    stride: int
    domain_yx: NDArray[np.uint16]
    dom_feat: NDArray[np.float32]


def _build_pool(
    img_gray: NDArray[np.float32],
    block: int,
    stride: int,
    *,
    max_domains: int | None,
    seed: int,
) -> _Pool:
    """
    Build a single pool for a given block size b.
    Domains are extracted from (2b x 2b), downsampled to (b x b), flattened, normalized.
    domain_yx generated via fastfractal.core.blocks.domains_yx (C-accelerated when available).
    """
    h, w = int(img_gray.shape[0]), int(img_gray.shape[1])

    yx = domains_yx(h, w, int(block), int(stride))
    if yx.shape[0] == 0:
        raise ValueError(f"domain pool empty for block={block} (need H,W >= 2*block)")

    if max_domains is not None and int(yx.shape[0]) > int(max_domains):
        rng = np.random.default_rng(int(seed) + int(block) * 101)
        keep = rng.choice(int(yx.shape[0]), size=int(max_domains), replace=False)
        keep.sort()
        yx = yx[keep].astype(np.uint16, copy=False)

    K = int(yx.shape[0])
    feats = np.empty((K, int(block) * int(block)), dtype=np.float32)

    for k in range(K):
        dy = int(yx[k, 0])
        dx = int(yx[k, 1])
        dom2 = img_gray[dy : dy + 2 * block, dx : dx + 2 * block]
        ds = downsample2x2(dom2)
        feats[k, :] = ds.reshape(-1).astype(np.float32, copy=False)

    feats = _normalize_rows(np.ascontiguousarray(feats, dtype=np.float32))

    return _Pool(
        block=int(block),
        stride=int(stride),
        domain_yx=np.ascontiguousarray(yx, dtype=np.uint16),
        dom_feat=feats,
    )


def _build_pools(
    img_gray: NDArray[np.float32],
    min_block: int,
    max_block: int,
    stride: int,
    *,
    max_domains: int | None,
    seed: int,
) -> list[_Pool]:
    pools: list[_Pool] = []
    b = int(min_block)
    while b <= int(max_block):
        pools.append(
            _build_pool(img_gray, b, stride, max_domains=max_domains, seed=seed)
        )
        b *= 2
    return pools


def _topk_select(
    range_vec_norm: NDArray[np.float32], dom_feat: NDArray[np.float32], topk: int
) -> NDArray[np.int32]:
    """
    range_vec_norm: (N,) normalized
    dom_feat: (K,N) normalized
    """
    K = int(dom_feat.shape[0])
    tk = int(max(1, min(int(topk), K)))

    r = np.ascontiguousarray(range_vec_norm.reshape(1, -1), dtype=np.float32)
    d = np.ascontiguousarray(dom_feat, dtype=np.float32)

    if _has_topk_dot():
        idx, _ = _cext_topk_dot(r, d, tk)
        return idx[0].astype(np.int32, copy=False)

    scores = d @ r[0]
    if tk >= int(scores.shape[0]):
        return np.argsort(-scores).astype(np.int32, copy=False)
    cand = np.argpartition(-scores, tk - 1)[:tk]
    cand = cand[np.argsort(-scores[cand])]
    return cand.astype(np.int32, copy=False)


_ACCEPTED_BACKENDS: set[str] = {
    "dot",
    "lsh",
    "pca_lsh",
    "pca",
    "pca_ann",
    "ann",
}


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

    min_block = int(min_block)
    max_block = int(max_block)
    stride = int(stride)

    if min_block <= 0 or max_block <= 0 or (max_block % min_block) != 0:
        raise ValueError("bad blocks")
    if (max_block & (max_block - 1)) != 0 or (min_block & (min_block - 1)) != 0:
        raise ValueError("blocks must be powers of two")

    backend = str(backend)
    if backend not in _ACCEPTED_BACKENDS:
        backend = "dot"

    if not use_quadtree:
        min_block = max_block

    img2, orig_h, orig_w = _pad_to_multiple(
        img.astype(np.float32, copy=False), max_block
    )
    h = int(img2.shape[0])
    w = int(img2.shape[1])

    c = 1 if img2.ndim == 2 else int(img2.shape[2])
    if c not in (1, 3):
        raise ValueError("channels must be 1 or 3")

    img_gray = _to_gray(img2)

    pools = _build_pools(
        img_gray,
        min_block=min_block,
        max_block=max_block,
        stride=stride,
        max_domains=max_domains,
        seed=int(seed),
    )
    pool_index = {p.block: i for i, p in enumerate(pools)}

    leaf_yx_list: list[tuple[int, int]] = []
    leaf_pool_list: list[int] = []
    leaf_dom_list: list[int] = []
    leaf_tf_list: list[int] = []
    leaf_codes_q_list: list[NDArray[np.uint8]] = []
    leaf_codes_f_list: list[NDArray[np.float32]] = []

    def emit_leaf(y0: int, x0: int, b0: int) -> float:
        pi = pool_index[int(b0)]
        p = pools[pi]

        r_proxy = (
            img_gray[y0 : y0 + b0, x0 : x0 + b0]
            .reshape(-1)
            .astype(np.float32, copy=False)
        )
        qv = _normalize_vec(r_proxy)

        cand = _topk_select(qv, p.dom_feat, int(topk))

        n_pix = int(b0) * int(b0)
        best_mse = float("inf")
        best_dom = 0
        best_tf = 0

        if quantized:
            best_q = np.zeros((c, 2), dtype=np.uint8)
        else:
            best_f = np.zeros((c, 2), dtype=np.float32)

        if c == 1:
            rflat0 = (
                img2[y0 : y0 + b0, x0 : x0 + b0]
                .reshape(-1)
                .astype(np.float32, copy=False)
            )
        else:
            rblk = img2[y0 : y0 + b0, x0 : x0 + b0, :].astype(np.float32, copy=False)
            rflat = np.transpose(rblk, (2, 0, 1)).reshape(c, -1)

        for k in cand:
            kk = int(k)
            dy = int(p.domain_yx[kk, 0])
            dx = int(p.domain_yx[kk, 1])

            if c == 1:
                dom2 = img2[dy : dy + 2 * b0, dx : dx + 2 * b0]
                ds = downsample2x2(dom2)

                for t in range(8):
                    dt = (
                        apply_transform_2d(ds, int(t))
                        .reshape(-1)
                        .astype(np.float32, copy=False)
                    )
                    s0, o0, _ = linreg_error(dt, rflat0)

                    s1 = float(np.clip(s0, -float(s_clip), float(s_clip)))
                    o1 = float(np.clip(o0, float(o_min), float(o_max)))

                    if quantized:
                        qs = _quant_s(s1, float(s_clip))
                        qo = _quant_o(o1, float(o_min), float(o_max))
                        s2 = _dequant_s(qs, float(s_clip))
                        o2 = _dequant_o(qo, float(o_min), float(o_max))
                        diff = (np.float32(s2) * dt + np.float32(o2) - rflat0).astype(
                            np.float32, copy=False
                        )
                        mse = float(np.dot(diff, diff) / float(n_pix))
                        if mse < best_mse:
                            best_mse = mse
                            best_dom = kk
                            best_tf = t
                            best_q[0, 0] = np.uint8(qs)
                            best_q[0, 1] = np.uint8(qo)
                    else:
                        diff = (np.float32(s1) * dt + np.float32(o1) - rflat0).astype(
                            np.float32, copy=False
                        )
                        mse = float(np.dot(diff, diff) / float(n_pix))
                        if mse < best_mse:
                            best_mse = mse
                            best_dom = kk
                            best_tf = t
                            best_f[0, 0] = np.float32(s1)
                            best_f[0, 1] = np.float32(o1)

            else:
                ds_ch: list[NDArray[np.float32]] = []
                for ch in range(c):
                    dom2c = img2[dy : dy + 2 * b0, dx : dx + 2 * b0, ch]
                    ds_ch.append(downsample2x2(dom2c))

                for t in range(8):
                    mse_sum = 0.0
                    if quantized:
                        qtmp = np.zeros((c, 2), dtype=np.uint8)
                    else:
                        ftmp = np.zeros((c, 2), dtype=np.float32)

                    for ch in range(c):
                        dt = (
                            apply_transform_2d(ds_ch[ch], int(t))
                            .reshape(-1)
                            .astype(np.float32, copy=False)
                        )
                        s0, o0, _ = linreg_error(dt, rflat[ch])

                        s1 = float(np.clip(s0, -float(s_clip), float(s_clip)))
                        o1 = float(np.clip(o0, float(o_min), float(o_max)))

                        if quantized:
                            qs = _quant_s(s1, float(s_clip))
                            qo = _quant_o(o1, float(o_min), float(o_max))
                            s2 = _dequant_s(qs, float(s_clip))
                            o2 = _dequant_o(qo, float(o_min), float(o_max))
                            diff = (
                                np.float32(s2) * dt + np.float32(o2) - rflat[ch]
                            ).astype(np.float32, copy=False)
                            mse_sum += float(np.dot(diff, diff) / float(n_pix))
                            qtmp[ch, 0] = np.uint8(qs)
                            qtmp[ch, 1] = np.uint8(qo)
                        else:
                            diff = (
                                np.float32(s1) * dt + np.float32(o1) - rflat[ch]
                            ).astype(np.float32, copy=False)
                            mse_sum += float(np.dot(diff, diff) / float(n_pix))
                            ftmp[ch, 0] = np.float32(s1)
                            ftmp[ch, 1] = np.float32(o1)

                    mse = mse_sum / float(c)
                    if mse < best_mse:
                        best_mse = mse
                        best_dom = kk
                        best_tf = t
                        if quantized:
                            best_q = qtmp
                        else:
                            best_f = ftmp

        leaf_yx_list.append((int(y0), int(x0)))
        leaf_pool_list.append(int(pi))
        leaf_dom_list.append(int(best_dom))
        leaf_tf_list.append(int(best_tf))

        if quantized:
            leaf_codes_q_list.append(best_q.astype(np.uint8, copy=False))
        else:
            leaf_codes_f_list.append(best_f.astype(np.float32, copy=False))

        return float(best_mse)

    def encode_node(y0: int, x0: int, block0: int) -> None:
        mse = emit_leaf(y0, x0, block0)
        if mse <= float(max_mse) or block0 <= int(min_block):
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
        b0 = int(max_block)
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


def encode_to_file(input_path: Path, output_path: Path, **kwargs: object) -> None:
    """
    File pipeline:
      - load_image(Path) -> float32 image in [0,1]
      - save_code(Path, FractalCode)
    """
    img = load_image(input_path)
    code = encode_array(img, **kwargs)
    save_code(output_path, code)


def encode(input_path: Path, output_path: Path, **kwargs: object) -> None:
    encode_to_file(input_path, output_path, **kwargs)
