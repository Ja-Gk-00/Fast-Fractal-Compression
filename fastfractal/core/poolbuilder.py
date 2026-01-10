from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from fastfractal import _cext  # type: ignore
from fastfractal.core.blocks import iter_domains
from fastfractal.core.quantization import rgb_to_luma
from fastfractal.core.search import (
    LSHIndex,
    SearchBackend,
    fit_pca,
    normalize_rows,
)
from fastfractal.core.transforms import apply_transform_2d
from fastfractal.core.types import PoolRuntime
from fastfractal.io.codebook import normalize_transform_ids
from fastfractal.utils.entropy import entropy01, var01

_HAS_DOWNSAMPLE = hasattr(_cext, "downsample2x2")


def _downsample2x2_f32(x: NDArray[np.float32]) -> NDArray[np.float32]:
    if _HAS_DOWNSAMPLE:
        return _cext.downsample2x2(x)  # type: ignore[attr-defined, unused-ignore, no-any-return]
    y = (x[0::2, 0::2] + x[1::2, 0::2] + x[0::2, 1::2] + x[1::2, 1::2]) * np.float32(
        0.25
    )
    return y.astype(np.float32, copy=False)


def bucket_id(ent: float, var: float, bucket_count: int) -> int:
    vn = _clip01(var / 0.25)
    s = 0.5 * _clip01(ent) + 0.5 * vn
    i = int(s * float(bucket_count))
    if i >= bucket_count:
        return bucket_count - 1
    if i < 0:
        return 0
    return i


def _clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


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
    transform_ids: str | tuple[int, ...] | None,
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

    tids = normalize_transform_ids(transform_ids)
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
