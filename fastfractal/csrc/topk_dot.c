#define FF_NO_IMPORT_ARRAY
#include "ff_common.h"

#include <math.h>
#include <string.h>


#ifdef I
#undef I
#endif


static inline void insert_topk(int topk, float v, int idx, float* vals, int32_t* inds) {
    int j = topk - 1;
    if (v <= vals[j]) return;
    while (j > 0 && v > vals[j - 1]) {
        vals[j] = vals[j - 1];
        inds[j] = inds[j - 1];
        j--;
    }
    vals[j] = v;
    inds[j] = (int32_t)idx;
}

PyObject* ff_topk_dot(PyObject* self, PyObject* args) {
    PyObject *ranges_obj, *domains_obj;
    int topk;
    if (!PyArg_ParseTuple(args, "OOi", &ranges_obj, &domains_obj, &topk)) return NULL;

    PyArrayObject* ranges = (PyArrayObject*)PyArray_FROM_OTF(
        ranges_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY
    );
    PyArrayObject* domains = (PyArrayObject*)PyArray_FROM_OTF(
        domains_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY
    );
    if (!ranges || !domains) { Py_XDECREF(ranges); Py_XDECREF(domains); return NULL; }

    if (PyArray_NDIM(ranges) != 2 || PyArray_NDIM(domains) != 2) {
        PyErr_SetString(PyExc_ValueError, "ranges/domains must be 2D");
        Py_DECREF(ranges); Py_DECREF(domains);
        return NULL;
    }

    npy_intp M = PyArray_DIM(ranges, 0);
    npy_intp N = PyArray_DIM(ranges, 1);
    npy_intp K = PyArray_DIM(domains, 0);
    if (PyArray_DIM(domains, 1) != N) {
        PyErr_SetString(PyExc_ValueError, "feature dim mismatch");
        Py_DECREF(ranges); Py_DECREF(domains);
        return NULL;
    }

    if (topk <= 0) { PyErr_SetString(PyExc_ValueError, "topk must be > 0"); Py_DECREF(ranges); Py_DECREF(domains); return NULL; }
    if (topk > (int)K) topk = (int)K;

    npy_intp dims[2] = {M, topk};
    PyArrayObject* out_idx = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_INT32);
    PyArrayObject* out_val = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_FLOAT32);
    if (!out_idx || !out_val) {
        Py_XDECREF(out_idx); Py_XDECREF(out_val);
        Py_DECREF(ranges); Py_DECREF(domains);
        return NULL;
    }

    const float* R = (const float*)PyArray_DATA(ranges);
    const float* D = (const float*)PyArray_DATA(domains);
    int32_t* outI = (int32_t*)PyArray_DATA(out_idx);
    float* outV = (float*)PyArray_DATA(out_val);

    for (npy_intp i = 0; i < M; i++) {
        float* rowV = outV + i * topk;
        int32_t* rowI = outI + i * topk;
        for (int t = 0; t < topk; t++) { rowV[t] = -1e30f; rowI[t] = -1; }

        const float* r = R + i * N;
        for (npy_intp k = 0; k < K; k++) {
            const float* d = D + k * N;
            double dot = 0.0;
            for (npy_intp j = 0; j < N; j++) dot += (double)r[j] * (double)d[j];
            insert_topk(topk, (float)dot, (int)k, rowV, rowI);
        }
    }

    Py_DECREF(ranges);
    Py_DECREF(domains);
    return Py_BuildValue("NN", (PyObject*)out_idx, (PyObject*)out_val);
}
