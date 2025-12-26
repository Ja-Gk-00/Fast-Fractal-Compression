from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray


def normalize_rows(x: NDArray[np.float32]) -> NDArray[np.float32]:
    mean = x.mean(axis=1, keepdims=True)
    xc = x - mean
    var = (xc * xc).mean(axis=1, keepdims=True)
    std = np.sqrt(var + np.float32(1e-12))
    z = xc / std
    nrm = np.sqrt((z * z).sum(axis=1, keepdims=True) + np.float32(1e-12))
    return z / nrm


def topk_from_subset(
    mat: NDArray[np.float32],
    q: NDArray[np.float32],
    subset: NDArray[np.int64],
    k: int,
) -> NDArray[np.int64]:
    if subset.size == 0:
        return subset
    scores = mat[subset] @ q
    if k >= scores.shape[0]:
        return subset[np.argsort(scores)[::-1]]
    idx = np.argpartition(scores, -k)[-k:]
    idx2 = idx[np.argsort(scores[idx])[::-1]]
    return subset[idx2]


def _sig_bits(vals: NDArray[np.float32]) -> int:
    bits = 0
    for i in range(vals.shape[0]):
        if float(vals[i]) > 0.0:
            bits |= 1 << i
    return bits


@dataclass(frozen=True, slots=True)
class PCAProjector:
    mean: NDArray[np.float32]
    basis: NDArray[np.float32]

    def project_matrix(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        xc = x - self.mean[None, :]
        return (xc @ self.basis.T).astype(np.float32, copy=False)

    def project_vector(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        xc = x - self.mean
        return (xc @ self.basis.T).astype(np.float32, copy=False)


def fit_pca(x: NDArray[np.float32], dim: int, sample: int, seed: int) -> PCAProjector:
    n, d = x.shape
    if dim >= d:
        mean = x.mean(axis=0).astype(np.float32, copy=False)
        basis = np.eye(d, dtype=np.float32)[:dim, :]
        return PCAProjector(mean=mean, basis=basis)

    rng = np.random.default_rng(seed)
    if n > sample:
        idx = rng.choice(n, size=sample, replace=False)
        xs = x[idx]
    else:
        xs = x

    mean = xs.mean(axis=0).astype(np.float32, copy=False)
    xc = (xs - mean[None, :]).astype(np.float32, copy=False)
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    basis = vt[:dim, :].astype(np.float32, copy=False)
    return PCAProjector(mean=mean, basis=basis)


@dataclass(frozen=True, slots=True)
class LSHIndex:
    proj: NDArray[np.float32]
    table: dict[int, NDArray[np.int64]]
    mat: NDArray[np.float32]

    def query(self, q: NDArray[np.float32], budget: int) -> NDArray[np.int64]:
        p = (self.proj @ q).astype(np.float32, copy=False)
        sig = _sig_bits(p)
        got = self.table.get(sig)
        if got is None:
            got = np.empty((0,), dtype=np.int64)
        if got.size >= budget:
            return got[:budget]
        acc = [got]
        need = budget - got.size
        planes = int(self.proj.shape[0])
        for i in range(planes):
            if need <= 0:
                break
            sig2 = sig ^ (1 << i)
            v = self.table.get(sig2)
            if v is None or v.size == 0:
                continue
            take = v if v.size <= need else v[:need]
            acc.append(take)
            need -= int(take.size)
        if len(acc) == 1:
            return acc[0]
        return np.concatenate(acc)

    @staticmethod
    def build(mat: NDArray[np.float32], planes: int, seed: int) -> LSHIndex:
        rng = np.random.default_rng(seed)
        proj = rng.standard_normal((planes, mat.shape[1]), dtype=np.float32)
        table_list: dict[int, list[int]] = {}
        for i in range(mat.shape[0]):
            sig = _sig_bits((proj @ mat[i]).astype(np.float32, copy=False))
            lst = table_list.get(sig)
            if lst is None:
                table_list[sig] = [i]
            else:
                lst.append(i)
        table: dict[int, NDArray[np.int64]] = {
            k: np.asarray(v, dtype=np.int64) for k, v in table_list.items()
        }
        return LSHIndex(proj=proj, table=table, mat=mat)


SearchBackend: Final[tuple[str, ...]] = ("dot", "lsh", "pca_lsh")
