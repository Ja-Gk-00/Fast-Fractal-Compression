#define FF_NO_IMPORT_ARRAY
#include "ff_common.h"


PyObject* ff_fit_so(PyObject* self, PyObject* args) {
    PyObject *r_obj, *d_obj;
    if (!PyArg_ParseTuple(args, "OO", &r_obj, &d_obj)) return NULL;

    PyArrayObject* r = (PyArrayObject*)PyArray_FROM_OTF(r_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* d = (PyArrayObject*)PyArray_FROM_OTF(d_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    if (!r || !d) { Py_XDECREF(r); Py_XDECREF(d); return NULL; }

    if (PyArray_NDIM(r) != 1 || PyArray_NDIM(d) != 1) {
        PyErr_SetString(PyExc_ValueError, "fit_so expects 1D float32 arrays");
        Py_DECREF(r); Py_DECREF(d); return NULL;
    }

    npy_intp n = PyArray_DIM(r, 0);
    if (PyArray_DIM(d, 0) != n) {
        PyErr_SetString(PyExc_ValueError, "fit_so expects same-length arrays");
        Py_DECREF(r); Py_DECREF(d); return NULL;
    }

    const float* rp = (const float*)PyArray_DATA(r);
    const float* dp = (const float*)PyArray_DATA(d);

    double sum_d = 0.0, sum_r = 0.0, sum_dd = 0.0, sum_dr = 0.0;
    for (npy_intp i = 0; i < n; i++) {
        double di = (double)dp[i];
        double ri = (double)rp[i];
        sum_d  += di;
        sum_r  += ri;
        sum_dd += di * di;
        sum_dr += di * ri;
    }

    double nn = (double)n;
    double denom = nn * sum_dd - sum_d * sum_d;
    double s = 0.0;
    if (denom != 0.0) {
        s = (nn * sum_dr - sum_d * sum_r) / denom;
    }
    double o = (sum_r - s * sum_d) / nn;

    double mse = 0.0;
    for (npy_intp i = 0; i < n; i++) {
        double pred = s * (double)dp[i] + o;
        double e = (double)rp[i] - pred;
        mse += e * e;
    }
    mse /= nn;

    Py_DECREF(r); Py_DECREF(d);
    return Py_BuildValue("ddd", s, o, mse);
}
