#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL FASTFRACTAL_ARRAY_API
#include <numpy/arrayobject.h>

#include <math.h>
#include <stdint.h>

PyObject* fastfractal_encode_leaf_best(PyObject* self, PyObject* args);

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

static inline double dot_f32(const float* a, const float* b, npy_intp n) {
    double acc = 0.0;
    npy_intp i = 0;
    for (; i + 3 < n; i += 4) {
        acc += (double)a[i] * (double)b[i];
        acc += (double)a[i + 1] * (double)b[i + 1];
        acc += (double)a[i + 2] * (double)b[i + 2];
        acc += (double)a[i + 3] * (double)b[i + 3];
    }
    for (; i < n; i++) acc += (double)a[i] * (double)b[i];
    return acc;
}

typedef struct {
    double score;
    npy_int64 idx;
} ScoreIdx;

static inline void heap_sift_down(ScoreIdx* heap, npy_intp n, npy_intp i) {
    while (1) {
        npy_intp l = 2 * i + 1;
        npy_intp r = l + 1;
        npy_intp smallest = i;

        if (l < n && heap[l].score < heap[smallest].score) smallest = l;
        if (r < n && heap[r].score < heap[smallest].score) smallest = r;
        if (smallest == i) break;

        ScoreIdx tmp = heap[i];
        heap[i] = heap[smallest];
        heap[smallest] = tmp;
        i = smallest;
    }
}

static inline void heap_build(ScoreIdx* heap, npy_intp n) {
    for (npy_intp i = (n / 2) - 1; i >= 0; i--) {
        heap_sift_down(heap, n, i);
        if (i == 0) break;
    }
}

static int cmp_desc_score(const void* a, const void* b) {
    const ScoreIdx* pa = (const ScoreIdx*)a;
    const ScoreIdx* pb = (const ScoreIdx*)b;
    if (pa->score < pb->score) return 1;
    if (pa->score > pb->score) return -1;
    return 0;
}

static PyObject* topk_from_subset(PyObject* self, PyObject* args) {
    PyObject* mat_obj = NULL;
    PyObject* q_obj = NULL;
    PyObject* subset_obj = NULL;
    int k_in = 0;

    if (!PyArg_ParseTuple(args, "OOOi", &mat_obj, &q_obj, &subset_obj, &k_in)) return NULL;

    PyArrayObject* mat = (PyArrayObject*)PyArray_FROM_OTF(mat_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    if (!mat) return NULL;

    PyArrayObject* q = (PyArrayObject*)PyArray_FROM_OTF(q_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    if (!q) { Py_DECREF(mat); return NULL; }

    PyArrayObject* subset = (PyArrayObject*)PyArray_FROM_OTF(subset_obj, NPY_NOTYPE, NPY_ARRAY_IN_ARRAY);
    if (!subset) { Py_DECREF(mat); Py_DECREF(q); return NULL; }

    if (PyArray_NDIM(mat) != 2) {
        PyErr_SetString(PyExc_ValueError, "mat must be 2D float32");
        Py_DECREF(mat); Py_DECREF(q); Py_DECREF(subset);
        return NULL;
    }
    if (PyArray_NDIM(q) != 1) {
        PyErr_SetString(PyExc_ValueError, "q must be 1D float32");
        Py_DECREF(mat); Py_DECREF(q); Py_DECREF(subset);
        return NULL;
    }
    if (PyArray_NDIM(subset) != 1) {
        PyErr_SetString(PyExc_ValueError, "subset must be 1D int32 or int64");
        Py_DECREF(mat); Py_DECREF(q); Py_DECREF(subset);
        return NULL;
    }

    const int subset_type = PyArray_TYPE(subset);
    if (!(subset_type == NPY_INT32 || subset_type == NPY_INT64)) {
        PyErr_SetString(PyExc_TypeError, "subset dtype must be int32 or int64");
        Py_DECREF(mat); Py_DECREF(q); Py_DECREF(subset);
        return NULL;
    }

    npy_intp rows = PyArray_DIM(mat, 0);
    npy_intp dim = PyArray_DIM(mat, 1);
    if (PyArray_DIM(q, 0) != dim) {
        PyErr_SetString(PyExc_ValueError, "q length must equal mat.shape[1]");
        Py_DECREF(mat); Py_DECREF(q); Py_DECREF(subset);
        return NULL;
    }

    npy_intp m = PyArray_DIM(subset, 0);
    if (m == 0 || k_in <= 0) {
        npy_intp odims[1] = {0};
        PyArrayObject* out0 = (PyArrayObject*)PyArray_SimpleNew(1, odims, NPY_INT64);
        Py_DECREF(mat); Py_DECREF(q); Py_DECREF(subset);
        return (PyObject*)out0;
    }

    npy_intp k = (npy_intp)k_in;
    if (k > m) k = m;

    float* matp = (float*)PyArray_DATA(mat);
    float* qp = (float*)PyArray_DATA(q);

    const int32_t* subp32 = NULL;
    const int64_t* subp64 = NULL;
    if (subset_type == NPY_INT32) subp32 = (const int32_t*)PyArray_DATA(subset);
    else subp64 = (const int64_t*)PyArray_DATA(subset);

    ScoreIdx* heap = (ScoreIdx*)PyMem_Malloc(sizeof(ScoreIdx) * (size_t)k);
    if (!heap) {
        PyErr_NoMemory();
        Py_DECREF(mat); Py_DECREF(q); Py_DECREF(subset);
        return NULL;
    }

    int error_flag = 0;

    Py_BEGIN_ALLOW_THREADS

    for (npy_intp i = 0; i < k; i++) {
        int64_t ridx64 = (subset_type == NPY_INT32) ? (int64_t)subp32[i] : (int64_t)subp64[i];
        if (ridx64 < 0 || ridx64 >= (int64_t)rows) { error_flag = 1; break; }

        const float* rowp = matp + ((npy_intp)ridx64) * dim;
        double s = dot_f32(rowp, qp, dim);
        heap[i].score = s;
        heap[i].idx = (npy_int64)ridx64;
    }

    if (!error_flag) {
        heap_build(heap, k);

        for (npy_intp i = k; i < m; i++) {
            int64_t ridx64 = (subset_type == NPY_INT32) ? (int64_t)subp32[i] : (int64_t)subp64[i];
            if (ridx64 < 0 || ridx64 >= (int64_t)rows) { error_flag = 1; break; }

            const float* rowp = matp + ((npy_intp)ridx64) * dim;
            double s = dot_f32(rowp, qp, dim);

            if (s > heap[0].score) {
                heap[0].score = s;
                heap[0].idx = (npy_int64)ridx64;
                heap_sift_down(heap, k, 0);
            }
        }
    }

    Py_END_ALLOW_THREADS

    if (error_flag) {
        PyMem_Free(heap);
        PyErr_SetString(PyExc_IndexError, "subset contains out-of-range row index");
        Py_DECREF(mat); Py_DECREF(q); Py_DECREF(subset);
        return NULL;
    }

    qsort(heap, (size_t)k, sizeof(ScoreIdx), cmp_desc_score);

    npy_intp odims[1] = {k};
    PyArrayObject* out = (PyArrayObject*)PyArray_SimpleNew(1, odims, NPY_INT64);
    if (!out) {
        PyMem_Free(heap);
        Py_DECREF(mat); Py_DECREF(q); Py_DECREF(subset);
        return NULL;
    }

    npy_int64* outp = (npy_int64*)PyArray_DATA(out);
    for (npy_intp i = 0; i < k; i++) outp[i] = heap[i].idx;

    PyMem_Free(heap);
    Py_DECREF(mat);
    Py_DECREF(q);
    Py_DECREF(subset);
    return (PyObject*)out;
}

static PyMethodDef Methods[] = {
    {"downsample2x2", (PyCFunction)downsample2x2, METH_VARARGS, NULL},
    {"linreg_error",  (PyCFunction)linreg_error,  METH_VARARGS, NULL},
    {"topk_from_subset", (PyCFunction)topk_from_subset, METH_VARARGS, NULL},
    {"encode_leaf_best", (PyCFunction)fastfractal_encode_leaf_best, METH_VARARGS, NULL},
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
    if (PyErr_Occurred()) return NULL;
    return PyModule_Create(&moduledef);
}
