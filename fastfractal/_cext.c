#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>

static PyObject* downsample2x2(PyObject* self, PyObject* args) {
    PyObject* src_obj = NULL;
    if (!PyArg_ParseTuple(args, "O", &src_obj)) return NULL;

    PyArrayObject* src = (PyArrayObject*)PyArray_FROM_OTF(src_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    if (!src) return NULL;

    if (PyArray_NDIM(src) != 2) {
        Py_DECREF(src);
        PyErr_SetString(PyExc_ValueError, "src must be 2D");
        return NULL;
    }

    npy_intp h = PyArray_DIM(src, 0);
    npy_intp w = PyArray_DIM(src, 1);
    if ((h % 2) != 0 || (w % 2) != 0) {
        Py_DECREF(src);
        PyErr_SetString(PyExc_ValueError, "src dims must be even");
        return NULL;
    }

    npy_intp oh = h / 2;
    npy_intp ow = w / 2;
    npy_intp dims[2] = {oh, ow};
    PyArrayObject* out = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_FLOAT32);
    if (!out) {
        Py_DECREF(src);
        return NULL;
    }

    float* s = (float*)PyArray_DATA(src);
    float* o = (float*)PyArray_DATA(out);
    npy_intp ss0 = PyArray_STRIDE(src, 0) / (npy_intp)sizeof(float);
    npy_intp ss1 = PyArray_STRIDE(src, 1) / (npy_intp)sizeof(float);

    for (npy_intp y = 0; y < oh; y++) {
        for (npy_intp x = 0; x < ow; x++) {
            npy_intp y0 = 2 * y;
            npy_intp x0 = 2 * x;
            float a = s[y0 * ss0 + x0 * ss1];
            float b = s[y0 * ss0 + (x0 + 1) * ss1];
            float c = s[(y0 + 1) * ss0 + x0 * ss1];
            float d = s[(y0 + 1) * ss0 + (x0 + 1) * ss1];
            o[y * ow + x] = 0.25f * (a + b + c + d);
        }
    }

    Py_DECREF(src);
    return (PyObject*)out;
}

static PyObject* linreg_error(PyObject* self, PyObject* args) {
    PyObject* d_obj = NULL;
    PyObject* r_obj = NULL;
    if (!PyArg_ParseTuple(args, "OO", &d_obj, &r_obj)) return NULL;

    PyArrayObject* d = (PyArrayObject*)PyArray_FROM_OTF(d_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    if (!d) return NULL;
    PyArrayObject* r = (PyArrayObject*)PyArray_FROM_OTF(r_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    if (!r) {
        Py_DECREF(d);
        return NULL;
    }

    if (PyArray_NDIM(d) != 1 || PyArray_NDIM(r) != 1) {
        Py_DECREF(d);
        Py_DECREF(r);
        PyErr_SetString(PyExc_ValueError, "inputs must be 1D");
        return NULL;
    }

    npy_intp n = PyArray_DIM(d, 0);
    if (PyArray_DIM(r, 0) != n) {
        Py_DECREF(d);
        Py_DECREF(r);
        PyErr_SetString(PyExc_ValueError, "length mismatch");
        return NULL;
    }

    float* dp = (float*)PyArray_DATA(d);
    float* rp = (float*)PyArray_DATA(r);

    double sumD = 0.0;
    double sumR = 0.0;
    double sumDD = 0.0;
    double sumRR = 0.0;
    double sumRD = 0.0;

    for (npy_intp i = 0; i < n; i++) {
        double dv = (double)dp[i];
        double rv = (double)rp[i];
        sumD += dv;
        sumR += rv;
        sumDD += dv * dv;
        sumRR += rv * rv;
        sumRD += dv * rv;
    }

    double dn = (double)n;
    double denom = dn * sumDD - sumD * sumD;

    double s = 0.0;
    double o = 0.0;

    if (fabs(denom) < 1e-18) {
        s = 0.0;
        o = sumR / dn;
    } else {
        s = (dn * sumRD - sumD * sumR) / denom;
        o = (sumR - s * sumD) / dn;
    }

    double err = sumRR + s * s * sumDD + dn * o * o - 2.0 * s * sumRD - 2.0 * o * sumR + 2.0 * s * o * sumD;

    Py_DECREF(d);
    Py_DECREF(r);

    return Py_BuildValue("ddd", s, o, err);
}

static PyMethodDef Methods[] = {
    {"downsample2x2", (PyCFunction)downsample2x2, METH_VARARGS, NULL},
    {"linreg_error", (PyCFunction)linreg_error, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_cext",
    NULL,
    -1,
    Methods
};

PyMODINIT_FUNC PyInit__cext(void) {
    import_array();
    return PyModule_Create(&moduledef);
}
