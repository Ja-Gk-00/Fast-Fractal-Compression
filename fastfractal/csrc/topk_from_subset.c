#define FF_NO_IMPORT_ARRAY
#include "ff_common.h"


static inline void insert_topk_i64(int topk, float v, int64_t idx, float* vals, int64_t* inds) {
    int j = topk - 1;
    if (v <= vals[j]) return;
    while (j > 0 && v > vals[j - 1]) {
        vals[j] = vals[j - 1];
        inds[j] = inds[j - 1];
        j--;
    }
    vals[j] = v;
    inds[j] = idx;
}

PyObject* ff_topk_from_subset(PyObject* self, PyObject* args) {
    PyObject *mat_obj, *q_obj, *subset_obj;
    int topk;
    if (!PyArg_ParseTuple(args, "OOOi", &mat_obj, &q_obj, &subset_obj, &topk)) return NULL;

    if (topk <= 0) {
        PyErr_SetString(PyExc_ValueError, "topk must be > 0");
        return NULL;
    }

    PyArrayObject* mat = (PyArrayObject*)PyArray_FROM_OTF(mat_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* q   = (PyArrayObject*)PyArray_FROM_OTF(q_obj,   NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* sub = (PyArrayObject*)PyArray_FROM_OTF(subset_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    if (!mat || !q || !sub) { Py_XDECREF(mat); Py_XDECREF(q); Py_XDECREF(sub); return NULL; }

    if (!ff_require_ndim(mat, 2, "mat must be 2D float32")) { Py_DECREF(mat); Py_DECREF(q); Py_DECREF(sub); return NULL; }
    if (PyArray_NDIM(q) != 1) { PyErr_SetString(PyExc_ValueError, "q must be 1D float32"); Py_DECREF(mat); Py_DECREF(q); Py_DECREF(sub); return NULL; }
    if (!ff_require_ndim(sub, 1, "subset must be 1D int64")) { Py_DECREF(mat); Py_DECREF(q); Py_DECREF(sub); return NULL; }

    npy_intp K = PyArray_DIM(mat, 0);
    npy_intp N = PyArray_DIM(mat, 1);
    if (PyArray_DIM(q, 0) != N) {
        PyErr_SetString(PyExc_ValueError, "mat and q must have same feature dimension");
        Py_DECREF(mat); Py_DECREF(q); Py_DECREF(sub); return NULL;
    }

    npy_intp S = PyArray_DIM(sub, 0);
    if (S <= 0) {
        
        npy_intp od[1] = {0};
        Py_DECREF(mat); Py_DECREF(q); Py_DECREF(sub);
        return (PyObject*)PyArray_SimpleNew(1, od, NPY_INT64);
    }

    if (topk > (int)S) topk = (int)S;

    npy_intp odims[1] = {topk};
    PyArrayObject* out = (PyArrayObject*)PyArray_SimpleNew(1, odims, NPY_INT64);
    if (!out) { Py_DECREF(mat); Py_DECREF(q); Py_DECREF(sub); return NULL; }

    const float* M = (const float*)PyArray_DATA(mat);
    const float* Q = (const float*)PyArray_DATA(q);
    const int64_t* SIDX = (const int64_t*)PyArray_DATA(sub);
    int64_t* O = (int64_t*)PyArray_DATA(out);

    
    float* vals = (float*)PyMem_Malloc((size_t)topk * sizeof(float));
    int64_t* inds = (int64_t*)PyMem_Malloc((size_t)topk * sizeof(int64_t));
    if (!vals || !inds) {
        PyMem_Free(vals); PyMem_Free(inds);
        Py_DECREF(out); Py_DECREF(mat); Py_DECREF(q); Py_DECREF(sub);
        PyErr_NoMemory();
        return NULL;
    }
    for (int t = 0; t < topk; t++) { vals[t] = -1e30f; inds[t] = -1; }

    for (npy_intp si = 0; si < S; si++) {
        int64_t ridx = SIDX[si];
        if (ridx < 0 || ridx >= (int64_t)K) continue;

        const float* row = M + (npy_intp)ridx * N;
        double dot = 0.0;
        for (npy_intp j = 0; j < N; j++) dot += (double)row[j] * (double)Q[j];
        insert_topk_i64(topk, (float)dot, ridx, vals, inds);
    }

    for (int t = 0; t < topk; t++) O[t] = inds[t];

    PyMem_Free(vals);
    PyMem_Free(inds);

    Py_DECREF(mat); Py_DECREF(q); Py_DECREF(sub);
    return (PyObject*)out;
}
