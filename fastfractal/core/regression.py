from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from fastfractal import _cext  # type: ignore

_HAS_LIN = hasattr(_cext, "linreg_error")
_HAS_QUAD = hasattr(_cext, "quadreg_error")
_HAS_HUBER = hasattr(_cext, "huber_error")
_HAS_CAUCHY = hasattr(_cext, "cauchy_error")

_HAS_SIG = hasattr(_cext, "sigmoid_error") or hasattr(_cext, "sigmoidreg_error")
_HAS_FAST = hasattr(_cext, "fast_error") or hasattr(_cext, "fastreg_error")

_HUBER_DELTA = 0.05
_CAUCHY_SCALE = 0.05
_RIDGE_LAMBDA_PER_N = 1e-4
_SIG_ALPHA = 12.0
_SIG_BETA = 0.05
_IRLS_ITERS = 2


def _as_f64_1d(x: NDArray[np.float32]) -> NDArray[np.float64]:
    return x.astype(np.float64, copy=False).ravel()


def _weighted_solve(
    sumW: float,
    sumWD: float,
    sumWDD: float,
    sumWR: float,
    sumWDR: float,
) -> tuple[float, float]:
    denom = sumW * sumWDD - sumWD * sumWD
    if abs(denom) < 1e-18 or sumW <= 1e-18:
        return 0.0, (sumWR / sumW) if sumW > 1e-18 else 0.0
    s = (sumW * sumWDR - sumWD * sumWR) / denom
    o = (sumWR - s * sumWD) / sumW
    return float(s), float(o)


def linreg_error(
    d: NDArray[np.float32], r: NDArray[np.float32]
) -> tuple[float, float, float]:
    if _HAS_LIN:
        s, o, e = _cext.linreg_error(d, r)
        return float(s), float(o), float(e)

    dv = d.astype(np.float64, copy=False).ravel()
    rv = r.astype(np.float64, copy=False).ravel()
    n = float(dv.size)

    sumD = float(dv.sum())
    sumR = float(rv.sum())
    sumDD = float(np.dot(dv, dv))
    sumRR = float(np.dot(rv, rv))
    sumRD = float(np.dot(dv, rv))

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


def quadreg_error(
    d: NDArray[np.float32], r: NDArray[np.float32]
) -> tuple[float, float, float]:
    if _HAS_QUAD:
        s, o, e = _cext.quadreg_error(d, r)
        return float(s), float(o), float(e)

    dv = d.astype(np.float64, copy=False).ravel()
    rv = r.astype(np.float64, copy=False).ravel()
    n = float(dv.size)

    sumD = float(dv.sum())
    sumR = float(rv.sum())
    sumDD = float(np.dot(dv, dv))
    sumRR = float(np.dot(rv, rv))
    sumRD = float(np.dot(dv, rv))

    lam = float(_RIDGE_LAMBDA_PER_N) * n
    denom = n * (sumDD + lam) - sumD * sumD
    if abs(denom) < 1e-18:
        s = 0.0
        o = sumR / n
    else:
        s = (n * sumRD - sumD * sumR) / denom
        o = ((sumDD + lam) * sumR - sumD * sumRD) / denom

    err = (
        sumRR
        + s * s * sumDD
        + n * o * o
        - 2.0 * s * sumRD
        - 2.0 * o * sumR
        + 2.0 * s * o * sumD
    )
    return float(s), float(o), float(err)


def huber_error(
    d: NDArray[np.float32], r: NDArray[np.float32]
) -> tuple[float, float, float]:
    if _HAS_HUBER:
        s, o, e = _cext.huber_error(d, r)
        return float(s), float(o), float(e)

    d1 = _as_f64_1d(d)
    r1 = _as_f64_1d(r)

    s, o, _ = linreg_error(d, r)

    for _ in range(_IRLS_ITERS):
        res = r1 - (s * d1 + o)
        a = np.abs(res)
        w = np.ones_like(a)
        m = a > _HUBER_DELTA
        w[m] = _HUBER_DELTA / a[m]

        sumW = float(w.sum())
        sumWD = float(np.dot(w, d1))
        sumWDD = float(np.dot(w, d1 * d1))
        sumWR = float(np.dot(w, r1))
        sumWDR = float(np.dot(w, d1 * r1))
        s, o = _weighted_solve(sumW, sumWD, sumWDD, sumWR, sumWDR)

    res = r1 - (s * d1 + o)
    err = float(np.dot(res, res))
    return float(s), float(o), float(err)


def cauchy_error(
    d: NDArray[np.float32], r: NDArray[np.float32]
) -> tuple[float, float, float]:
    if _HAS_CAUCHY:
        s, o, e = _cext.cauchy_error(d, r)
        return float(s), float(o), float(e)

    d1 = _as_f64_1d(d)
    r1 = _as_f64_1d(r)
    inv_sc2 = 1.0 / (_CAUCHY_SCALE * _CAUCHY_SCALE)

    s, o, _ = linreg_error(d, r)

    for _ in range(_IRLS_ITERS):
        res = r1 - (s * d1 + o)
        w = 1.0 / (1.0 + (res * res) * inv_sc2)

        sumW = float(w.sum())
        sumWD = float(np.dot(w, d1))
        sumWDD = float(np.dot(w, d1 * d1))
        sumWR = float(np.dot(w, r1))
        sumWDR = float(np.dot(w, d1 * r1))
        s, o = _weighted_solve(sumW, sumWD, sumWDD, sumWR, sumWDR)

    res = r1 - (s * d1 + o)
    err = float(np.dot(res, res))
    return float(s), float(o), float(err)


def sigmoid_error(
    d: NDArray[np.float32], r: NDArray[np.float32]
) -> tuple[float, float, float]:
    if _HAS_SIG:
        if hasattr(_cext, "sigmoid_error"):
            s, o, e = _cext.sigmoid_error(d, r)
        else:
            s, o, e = _cext.sigmoidreg_error(d, r)
        return float(s), float(o), float(e)

    d1 = _as_f64_1d(d)
    r1 = _as_f64_1d(r)

    s, o, _ = linreg_error(d, r)

    for _ in range(_IRLS_ITERS):
        res = r1 - (s * d1 + o)
        a = np.abs(res)
        w = 1.0 / (1.0 + np.exp(_SIG_ALPHA * (a - _SIG_BETA)))

        sumW = float(w.sum())
        sumWD = float(np.dot(w, d1))
        sumWDD = float(np.dot(w, d1 * d1))
        sumWR = float(np.dot(w, r1))
        sumWDR = float(np.dot(w, d1 * r1))
        s, o = _weighted_solve(sumW, sumWD, sumWDD, sumWR, sumWDR)

    res = r1 - (s * d1 + o)
    err = float(np.dot(res, res))
    return float(s), float(o), float(err)


def sigmoidreg_error(
    d: NDArray[np.float32], r: NDArray[np.float32]
) -> tuple[float, float, float]:
    return sigmoid_error(d, r)


def fastreg_error(
    d: NDArray[np.float32], r: NDArray[np.float32]
) -> tuple[float, float, float]:
    if _HAS_FAST:
        if hasattr(_cext, "fast_error"):
            s, o, e = _cext.fast_error(d, r)
        else:
            s, o, e = _cext.fastreg_error(d, r)
        return float(s), float(o), float(e)

    rv = r.astype(np.float64, copy=False).ravel()
    n = float(rv.size)
    o = float(rv.sum()) / n
    res = rv - o
    err = float(np.dot(res, res))
    return 0.0, float(o), float(err)


RegressionKind = Literal["linear", "quadreg", "sigmoid", "fast", "huber", "cauchy"]

_REG_ALIASES: dict[str, RegressionKind] = {
    "linear": "linear",
    "lin": "linear",
    "ls": "linear",
    "least_squares": "linear",
    "quad": "quadreg",
    "quadratic": "quadreg",
    "quadreg": "quadreg",
    "ridge": "quadreg",
    "sigmoid": "sigmoid",
    "sigmoidal": "sigmoid",
    "logistic": "sigmoid",
    "fast": "fast",
    "mean": "fast",
    "huber": "huber",
    "cauchy": "cauchy",
}

_REG_ID: dict[RegressionKind, int] = {
    "linear": 0,
    "quadreg": 1,
    "sigmoid": 2,
    "fast": 3,
    "huber": 4,
    "cauchy": 5,
}

RegressionFn = Callable[
    [NDArray[np.float32], NDArray[np.float32]], tuple[float, float, float]
]


def normalize_regression_name(name: str) -> RegressionKind:
    key = name.strip().lower()
    if key in _REG_ALIASES:
        return _REG_ALIASES[key]
    raise ValueError(f"unknown regression: {name!r}")


def regression_id(regression: str | int | RegressionKind | RegressionFn | None) -> int:
    if regression is None:
        return _REG_ID["linear"]
    if isinstance(regression, int):
        if 0 <= regression <= 5:
            return regression
        raise ValueError("regression id out of range (expected 0..5)")
    if isinstance(regression, str):
        return _REG_ID[normalize_regression_name(regression)]
    return _REG_ID["linear"]


def resolve_regression(
    regression: str | RegressionKind | RegressionFn | None,
) -> RegressionFn:
    if regression is None:
        return linreg_error
    if callable(regression):
        return regression
    if isinstance(regression, str):
        kind = normalize_regression_name(regression)
    else:
        kind = regression

    if kind == "linear":
        return linreg_error
    if kind == "quadreg":
        return quadreg_error
    if kind == "sigmoid":
        return sigmoid_error
    if kind == "fast":
        return fastreg_error
    if kind == "huber":
        return huber_error
    if kind == "cauchy":
        return cauchy_error
    raise ValueError(f"unknown regression kind: {kind!r}")
