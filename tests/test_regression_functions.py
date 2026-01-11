from __future__ import annotations

import numpy as np
import pytest

from fastfractal.core.regression import (
    cauchy_error,
    fastreg_error,
    huber_error,
    linreg_error,
    normalize_regression_name,
    quadreg_error,
    regression_id,
    resolve_regression,
    sigmoid_error,
)


def _affine_data(
    n: int = 256,
    s_true: float = 0.7,
    o_true: float = -0.2,
    noise: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    rng = np.random.default_rng(seed)
    d = rng.uniform(-1.0, 1.0, size=n).astype(np.float32)
    r = (s_true * d + o_true).astype(np.float32)
    if noise > 0:
        r = (r + rng.normal(0.0, noise, size=n).astype(np.float32)).astype(np.float32)
    return d, r, float(s_true), float(o_true)


def _inlier_outlier_data(
    n: int = 512,
    outliers: int = 16,
    s_true: float = 0.65,
    o_true: float = 0.05,
    noise: float = 0.01,
    outlier_scale: float = 4.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    d = rng.uniform(-1.0, 1.0, size=n).astype(np.float32)
    r = (s_true * d + o_true).astype(np.float32)
    r = (r + rng.normal(0.0, noise, size=n).astype(np.float32)).astype(np.float32)

    mask_inlier = np.ones(n, dtype=bool)
    idx = rng.choice(n, size=outliers, replace=False)
    mask_inlier[idx] = False

    r[idx] = (r[idx] + rng.normal(0.0, outlier_scale, size=outliers)).astype(np.float32)
    return d, r, mask_inlier


def _pred(d: np.ndarray, s: float, o: float) -> np.ndarray:
    return (s * d.astype(np.float64) + o).astype(np.float64)


def test_normalize_and_resolve_regression() -> None:
    assert normalize_regression_name("lin") == "linear"
    assert normalize_regression_name("ridge") == "quadreg"
    assert normalize_regression_name("logistic") == "sigmoid"
    assert normalize_regression_name("mean") == "fast"

    assert regression_id(None) == 0
    assert regression_id("linear") == 0
    assert regression_id("quadreg") == 1
    assert regression_id("sigmoid") == 2
    assert regression_id("fast") == 3
    assert regression_id("huber") == 4
    assert regression_id("cauchy") == 5

    assert resolve_regression("linear") is linreg_error
    assert resolve_regression("quadreg") is quadreg_error
    assert resolve_regression("sigmoid") is sigmoid_error
    assert resolve_regression("fast") is fastreg_error
    assert resolve_regression("huber") is huber_error
    assert resolve_regression("cauchy") is cauchy_error

    with pytest.raises(ValueError):
        normalize_regression_name("unknown_regression_kind")


def test_linreg_recovers_exact_affine() -> None:
    d, r, s_true, o_true = _affine_data(
        n=1024, s_true=0.8, o_true=-0.15, noise=0.0, seed=1
    )
    s, o, e = linreg_error(d, r)

    assert np.isfinite([s, o, e]).all()
    assert abs(s - s_true) < 1e-6
    assert abs(o - o_true) < 1e-6
    assert e < 1e-10


def test_huber_cauchy_sigmoid_match_lin_on_clean_data() -> None:
    d, r, s_true, o_true = _affine_data(
        n=1024, s_true=-0.6, o_true=0.25, noise=0.0, seed=2
    )

    s0, o0, e0 = linreg_error(d, r)
    assert abs(s0 - s_true) < 1e-6
    assert abs(o0 - o_true) < 1e-6
    assert e0 < 1e-10

    for fn in (huber_error, cauchy_error, sigmoid_error):
        s, o, e = fn(d, r)
        assert np.isfinite([s, o, e]).all()
        assert abs(s - s_true) < 1e-6
        assert abs(o - o_true) < 1e-6
        assert e < 1e-8


def test_fastreg_returns_mean_only() -> None:
    rng = np.random.default_rng(3)
    d = rng.uniform(-1.0, 1.0, size=2048).astype(np.float32)
    r = rng.uniform(-0.5, 1.5, size=2048).astype(np.float32)

    s, o, e = fastreg_error(d, r)
    assert np.isfinite([s, o, e]).all()
    assert s == 0.0

    o_exp = float(r.astype(np.float64).mean())
    assert abs(o - o_exp) < 1e-12

    res = r.astype(np.float64) - o_exp
    e_exp = float(np.dot(res, res))
    assert abs(e - e_exp) < 1e-8


def test_quadreg_constant_domain_reduces_to_mean_only() -> None:
    rng = np.random.default_rng(4)
    d = np.full(1024, 0.37, dtype=np.float32)
    r = rng.uniform(-0.5, 1.5, size=1024).astype(np.float32)

    s, o, e = quadreg_error(d, r)
    assert np.isfinite([s, o, e]).all()

    assert abs(s - 0.0) < 1e-8
    o_exp = float(r.astype(np.float64).mean())
    assert abs(o - o_exp) < 1e-10

    res = r.astype(np.float64) - o_exp
    e_exp = float(np.dot(res, res))
    assert abs(e - e_exp) < 1e-6


@pytest.mark.parametrize("robust_fn", [huber_error, cauchy_error, sigmoid_error])
def test_robust_regressions_reduce_inlier_mse_vs_linear(robust_fn) -> None:
    d, r, inlier = _inlier_outlier_data(
        n=1024, outliers=32, noise=0.01, outlier_scale=6.0, seed=5
    )

    s_lin, o_lin, _ = linreg_error(d, r)
    s_rb, o_rb, _ = robust_fn(d, r)

    pred_lin = _pred(d, s_lin, o_lin)
    pred_rb = _pred(d, s_rb, o_rb)
    rr = r.astype(np.float64)

    mse_lin_in = float(np.mean((rr[inlier] - pred_lin[inlier]) ** 2))
    mse_rb_in = float(np.mean((rr[inlier] - pred_rb[inlier]) ** 2))

    assert np.isfinite([mse_lin_in, mse_rb_in]).all()
    assert mse_rb_in < mse_lin_in
