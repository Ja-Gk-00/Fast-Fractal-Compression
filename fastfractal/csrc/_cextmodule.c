#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define FASTFRACTAL_IMPORT_ARRAY 1
#include "ff_common.h"


#include <Python.h>
#include <numpy/arrayobject.h>

#include <stdint.h>
#include <math.h>
#include <string.h>


   
static int ff_require_float32(PyArrayObject* arr, const char* msg) {
    if (PyArray_TYPE(arr) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_TypeError, msg);
        return 0;
    }
    return 1;
}

static int ff_require_contig(PyArrayObject* arr, const char* msg) {
    if (!PyArray_ISCARRAY(arr)) {
        PyErr_SetString(PyExc_ValueError, msg);
        return 0;
    }
    return 1;
}

static inline void ff_u16_yx(uint16_t* out, int idx, uint16_t y, uint16_t x) {
    out[2 * idx + 0] = y;
    out[2 * idx + 1] = x;
}


static inline void ff_insert_topk(float v, npy_intp idx, float* vals, npy_intp* ids, int k) {
    int j = k - 1;
    if (v <= vals[j]) return;
    while (j > 0 && v > vals[j - 1]) {
        vals[j] = vals[j - 1];
        ids[j] = ids[j - 1];
        j--;
    }
    vals[j] = v;
    ids[j] = idx;
}

static inline void ff_init_topk(float* vals, npy_intp* ids, int k) {
    for (int i = 0; i < k; i++) {
        vals[i] = -INFINITY;
        ids[i] = -1;
    }
}



static PyObject* ff_domains_yx(PyObject* self, PyObject* args) {
    int h, w, block, stride;
    if (!PyArg_ParseTuple(args, "iiii", &h, &w, &block, &stride)) return NULL;

    if (block <= 0) {
        PyErr_SetString(PyExc_ValueError, "block must be > 0");
        return NULL;
    }
    if (stride <= 0) {
        PyErr_SetString(PyExc_ValueError, "stride must be > 0");
        return NULL;
    }

    int D = 2 * block;
    int y_max = h - D;
    int x_max = w - D;
    if (y_max < 0 || x_max < 0) {
        npy_intp dims0[2] = {0, 2};
        return PyArray_SimpleNew(2, dims0, NPY_UINT16);
    }

    int ny = y_max / stride + 1;
    int nx = x_max / stride + 1;
    int n = ny * nx;

    npy_intp dims[2] = {(npy_intp)n, 2};
    PyArrayObject* arr = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_UINT16);
    if (!arr) return NULL;

    uint16_t* out = (uint16_t*)PyArray_DATA(arr);
    int idx = 0;
    for (int y = 0; y <= y_max; y += stride) {
        for (int x = 0; x <= x_max; x += stride) {
            ff_u16_yx(out, idx, (uint16_t)y, (uint16_t)x);
            idx++;
        }
    }

    return (PyObject*)arr;
}



static PyObject* ff_ranges_yx(PyObject* self, PyObject* args) {
    int h, w, block;
    if (!PyArg_ParseTuple(args, "iii", &h, &w, &block)) return NULL;

    if (block <= 0) {
        PyErr_SetString(PyExc_ValueError, "block must be > 0");
        return NULL;
    }

    int y_max = h - block;
    int x_max = w - block;
    if (y_max < 0 || x_max < 0) {
        npy_intp dims0[2] = {0, 2};
        return PyArray_SimpleNew(2, dims0, NPY_UINT16);
    }

    int n = (y_max + 1) * (x_max + 1);
    npy_intp dims[2] = {(npy_intp)n, 2};
    PyArrayObject* arr = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_UINT16);
    if (!arr) return NULL;

    uint16_t* out = (uint16_t*)PyArray_DATA(arr);
    int idx = 0;
    for (int y = 0; y <= y_max; y++) {
        for (int x = 0; x <= x_max; x++) {
            ff_u16_yx(out, idx, (uint16_t)y, (uint16_t)x);
            idx++;
        }
    }
    return (PyObject*)arr;
}



static PyObject* ff_extract_range(PyObject* self, PyObject* args) {
    PyObject* img_obj = NULL;
    int y, x, b;
    if (!PyArg_ParseTuple(args, "Oiii", &img_obj, &y, &x, &b)) return NULL;

    PyArrayObject* img = (PyArrayObject*)PyArray_FROM_OTF(img_obj, NPY_FLOAT32, NPY_ARRAY_ALIGNED);
    if (!img) return NULL;

    int ndim = PyArray_NDIM(img);
    if (!(ndim == 2 || ndim == 3)) {
        PyErr_SetString(PyExc_ValueError, "extract_range expects 2D or 3D float32 array");
        Py_DECREF(img);
        return NULL;
    }
    if (b <= 0) {
        PyErr_SetString(PyExc_ValueError, "block must be > 0");
        Py_DECREF(img);
        return NULL;
    }

    npy_intp H = PyArray_DIM(img, 0);
    npy_intp W = PyArray_DIM(img, 1);
    if (y < 0 || x < 0 || (npy_intp)(y + b) > H || (npy_intp)(x + b) > W) {
        PyErr_SetString(PyExc_ValueError, "extract_range: out of bounds");
        Py_DECREF(img);
        return NULL;
    }

    npy_intp dims[3] = {b, b, 0};
    if (ndim == 2) {
        PyArrayObject* out = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_FLOAT32);
        if (!out) { Py_DECREF(img); return NULL; }

        char* src0 = (char*)PyArray_DATA(img);
        npy_intp s0 = PyArray_STRIDE(img, 0);
        npy_intp s1 = PyArray_STRIDE(img, 1);

        float* dst = (float*)PyArray_DATA(out);
        for (int i = 0; i < b; i++) {
            char* row = src0 + (npy_intp)(y + i) * s0 + (npy_intp)x * s1;
            if (s1 == (npy_intp)sizeof(float)) {
                memcpy(dst + (npy_intp)i * b, row, (size_t)b * sizeof(float));
            } else {
                for (int j = 0; j < b; j++) {
                    dst[(npy_intp)i * b + j] = *(float*)(row + (npy_intp)j * s1);
                }
            }
        }

        Py_DECREF(img);
        return (PyObject*)out;
    }

    npy_intp C = PyArray_DIM(img, 2);
    dims[2] = C;
    PyArrayObject* out = (PyArrayObject*)PyArray_SimpleNew(3, dims, NPY_FLOAT32);
    if (!out) { Py_DECREF(img); return NULL; }

    char* src0 = (char*)PyArray_DATA(img);
    npy_intp s0 = PyArray_STRIDE(img, 0);
    npy_intp s1 = PyArray_STRIDE(img, 1);
    npy_intp s2 = PyArray_STRIDE(img, 2);

    float* dst = (float*)PyArray_DATA(out);
    npy_intp dst_s0 = PyArray_STRIDE(out, 0);
    npy_intp dst_s1 = PyArray_STRIDE(out, 1);
    npy_intp dst_s2 = PyArray_STRIDE(out, 2);

    for (int i = 0; i < b; i++) {
        for (int j = 0; j < b; j++) {
            char* pix = src0 + (npy_intp)(y + i) * s0 + (npy_intp)(x + j) * s1;
            char* opix = (char*)dst + (npy_intp)i * dst_s0 + (npy_intp)j * dst_s1;
            for (npy_intp k = 0; k < C; k++) {
                *(float*)(opix + k * dst_s2) = *(float*)(pix + k * s2);
            }
        }
    }

    Py_DECREF(img);
    return (PyObject*)out;
}



static PyObject* ff_extract_range_flat(PyObject* self, PyObject* args) {
    PyObject* img_obj = NULL;
    int y, x, b;
    if (!PyArg_ParseTuple(args, "Oiii", &img_obj, &y, &x, &b)) return NULL;

    PyArrayObject* img = (PyArrayObject*)PyArray_FROM_OTF(img_obj, NPY_FLOAT32, NPY_ARRAY_ALIGNED);
    if (!img) return NULL;

    if (!ff_require_ndim(img, 2, "extract_range_flat expects 2D float32 array")) {
        Py_DECREF(img);
        return NULL;
    }
    if (b <= 0) {
        PyErr_SetString(PyExc_ValueError, "block must be > 0");
        Py_DECREF(img);
        return NULL;
    }

    npy_intp H = PyArray_DIM(img, 0);
    npy_intp W = PyArray_DIM(img, 1);
    if (y < 0 || x < 0 || (npy_intp)(y + b) > H || (npy_intp)(x + b) > W) {
        PyErr_SetString(PyExc_ValueError, "extract_range_flat: out of bounds");
        Py_DECREF(img);
        return NULL;
    }

    npy_intp n = (npy_intp)b * (npy_intp)b;
    npy_intp dims[1] = {n};
    PyArrayObject* out = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_FLOAT32);
    if (!out) { Py_DECREF(img); return NULL; }

    float* dst = (float*)PyArray_DATA(out);
    char* src0 = (char*)PyArray_DATA(img);
    npy_intp s0 = PyArray_STRIDE(img, 0);
    npy_intp s1 = PyArray_STRIDE(img, 1);

    for (int i = 0; i < b; i++) {
        char* row = src0 + (npy_intp)(y + i) * s0 + (npy_intp)x * s1;
        if (s1 == (npy_intp)sizeof(float)) {
            memcpy(dst + (npy_intp)i * b, row, (size_t)b * sizeof(float));
        } else {
            for (int j = 0; j < b; j++) {
                dst[(npy_intp)i * b + j] = *(float*)(row + (npy_intp)j * s1);
            }
        }
    }

    Py_DECREF(img);
    return (PyObject*)out;
}



static PyObject* ff_topk_dot(PyObject* self, PyObject* args) {
    PyObject* mat_obj = NULL;
    PyObject* q_obj = NULL;
    int topk;
    if (!PyArg_ParseTuple(args, "OOi", &mat_obj, &q_obj, &topk)) return NULL;

    if (topk <= 0) {
        PyErr_SetString(PyExc_ValueError, "topk must be > 0");
        return NULL;
    }

    PyArrayObject* mat = (PyArrayObject*)PyArray_FROM_OTF(mat_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    if (!mat) return NULL;
    PyArrayObject* q = (PyArrayObject*)PyArray_FROM_OTF(q_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    if (!q) { Py_DECREF(mat); return NULL; }

    if (!ff_require_ndim(mat, 2, "topk_dot: mat must be 2D float32")) goto fail;
    if (!ff_require_ndim(q, 1, "topk_dot: q must be 1D float32")) goto fail;

    npy_intp n = PyArray_DIM(mat, 0);
    npy_intp d = PyArray_DIM(mat, 1);
    if (PyArray_DIM(q, 0) != d) {
        PyErr_SetString(PyExc_ValueError, "topk_dot: q length must match mat second dimension");
        goto fail;
    }
    if (topk > (int)n) topk = (int)n;

    npy_intp out_dims[1] = {topk};
    PyArrayObject* out = (PyArrayObject*)PyArray_SimpleNew(1, out_dims, NPY_INT64);
    if (!out) goto fail;

    float* best_vals = (float*)PyMem_Malloc((size_t)topk * sizeof(float));
    npy_intp* best_idx = (npy_intp*)PyMem_Malloc((size_t)topk * sizeof(npy_intp));
    if (!best_vals || !best_idx) {
        PyErr_NoMemory();
        Py_XDECREF(out);
        PyMem_Free(best_vals);
        PyMem_Free(best_idx);
        goto fail;
    }
    ff_init_topk(best_vals, best_idx, topk);

    char* mat_data = (char*)PyArray_DATA(mat);
    npy_intp s0 = PyArray_STRIDE(mat, 0);
    npy_intp s1 = PyArray_STRIDE(mat, 1);
    char* q_data = (char*)PyArray_DATA(q);
    npy_intp qs = PyArray_STRIDE(q, 0);

    for (npy_intp i = 0; i < n; i++) {
        float dot = 0.0f;
        char* row = mat_data + i * s0;
        for (npy_intp j = 0; j < d; j++) {
            float a = *(float*)(row + j * s1);
            float b = *(float*)(q_data + j * qs);
            dot += a * b;
        }
        ff_insert_topk(dot, i, best_vals, best_idx, topk);
    }

    int64_t* outp = (int64_t*)PyArray_DATA(out);
    for (int i = 0; i < topk; i++) outp[i] = (int64_t)best_idx[i];

    PyMem_Free(best_vals);
    PyMem_Free(best_idx);
    Py_DECREF(mat);
    Py_DECREF(q);
    return (PyObject*)out;

fail:
    Py_DECREF(mat);
    Py_DECREF(q);
    return NULL;
}



static PyObject* ff_topk_from_subset(PyObject* self, PyObject* args) {
    PyObject *mat_obj = NULL, *q_obj = NULL, *subset_obj = NULL;
    int topk;

    if (!PyArg_ParseTuple(args, "OOOi", &mat_obj, &q_obj, &subset_obj, &topk)) return NULL;

    if (topk <= 0) {
        PyErr_SetString(PyExc_ValueError, "topk must be > 0");
        return NULL;
    }

    PyArrayObject* mat = (PyArrayObject*)PyArray_FROM_OTF(mat_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    if (!mat) return NULL;
    PyArrayObject* q = (PyArrayObject*)PyArray_FROM_OTF(q_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    if (!q) { Py_DECREF(mat); return NULL; }
    PyArrayObject* subset = (PyArrayObject*)PyArray_FROM_OTF(subset_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    if (!subset) { Py_DECREF(mat); Py_DECREF(q); return NULL; }

    if (!ff_require_ndim(mat, 2, "topk_from_subset: mat must be 2D float32")) goto fail2;
    if (!ff_require_ndim(q, 1, "topk_from_subset: q must be 1D float32")) goto fail2;
    if (!ff_require_ndim(subset, 1, "topk_from_subset: subset must be 1D int32")) goto fail2;

    npy_intp n = PyArray_DIM(mat, 0);
    npy_intp d = PyArray_DIM(mat, 1);
    npy_intp m = PyArray_DIM(subset, 0);

    if (PyArray_DIM(q, 0) != d) {
        PyErr_SetString(PyExc_ValueError, "topk_from_subset: q length must match mat second dimension");
        goto fail2;
    }
    if (topk > (int)m) topk = (int)m;

    npy_intp out_dims[1] = {topk};
    PyArrayObject* out = (PyArrayObject*)PyArray_SimpleNew(1, out_dims, NPY_INT64);
    if (!out) goto fail2;

    float* best_vals = (float*)PyMem_Malloc((size_t)topk * sizeof(float));
    npy_intp* best_ids = (npy_intp*)PyMem_Malloc((size_t)topk * sizeof(npy_intp));
    if (!best_vals || !best_ids) {
        PyErr_NoMemory();
        Py_XDECREF(out);
        PyMem_Free(best_vals);
        PyMem_Free(best_ids);
        goto fail2;
    }
    ff_init_topk(best_vals, best_ids, topk);

    char* mat_data = (char*)PyArray_DATA(mat);
    npy_intp s0 = PyArray_STRIDE(mat, 0);
    npy_intp s1 = PyArray_STRIDE(mat, 1);

    char* q_data = (char*)PyArray_DATA(q);
    npy_intp qs = PyArray_STRIDE(q, 0);

    int32_t* sub = (int32_t*)PyArray_DATA(subset);

    for (npy_intp ii = 0; ii < m; ii++) {
        int32_t row_id = sub[ii];
        if (row_id < 0 || (npy_intp)row_id >= n) continue;

        char* row = mat_data + (npy_intp)row_id * s0;
        float dot = 0.0f;
        for (npy_intp j = 0; j < d; j++) {
            float a = *(float*)(row + j * s1);
            float b = *(float*)(q_data + j * qs);
            dot += a * b;
        }
        ff_insert_topk(dot, (npy_intp)row_id, best_vals, best_ids, topk);
    }

    int64_t* outp = (int64_t*)PyArray_DATA(out);
    for (int i = 0; i < topk; i++) outp[i] = (int64_t)best_ids[i];

    PyMem_Free(best_vals);
    PyMem_Free(best_ids);
    Py_DECREF(mat);
    Py_DECREF(q);
    Py_DECREF(subset);
    return (PyObject*)out;

fail2:
    Py_DECREF(mat);
    Py_DECREF(q);
    Py_DECREF(subset);
    return NULL;
}



static PyObject* ff_downsample2x2(PyObject* self, PyObject* args) {
    PyObject* dom_obj;
    if (!PyArg_ParseTuple(args, "O", &dom_obj)) return NULL;

    PyArrayObject* dom = (PyArrayObject*)PyArray_FROM_OTF(dom_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    if (!dom) return NULL;

    int ndim = PyArray_NDIM(dom);
    if (!(ndim == 2 || ndim == 3)) {
        PyErr_SetString(PyExc_ValueError, "downsample2x2 expects 2D or 3D float32 array");
        Py_DECREF(dom);
        return NULL;
    }

    npy_intp h = PyArray_DIM(dom, 0);
    npy_intp w = PyArray_DIM(dom, 1);
    if ((h % 2) || (w % 2)) {
        PyErr_SetString(PyExc_ValueError, "downsample2x2 expects even h,w");
        Py_DECREF(dom);
        return NULL;
    }

    npy_intp oh = h / 2;
    npy_intp ow = w / 2;

    if (ndim == 2) {
        npy_intp dims[2] = {oh, ow};
        PyArrayObject* out = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_FLOAT32);
        if (!out) { Py_DECREF(dom); return NULL; }

        float* src = (float*)PyArray_DATA(dom);
        float* dst = (float*)PyArray_DATA(out);

        npy_intp s0 = PyArray_STRIDE(dom, 0) / (npy_intp)sizeof(float);
        npy_intp s1 = PyArray_STRIDE(dom, 1) / (npy_intp)sizeof(float);

        for (npy_intp y = 0; y < oh; y++) {
            for (npy_intp x = 0; x < ow; x++) {
                npy_intp y2 = 2 * y;
                npy_intp x2 = 2 * x;
                float a = src[y2 * s0 + x2 * s1];
                float b = src[(y2 + 1) * s0 + x2 * s1];
                float c = src[y2 * s0 + (x2 + 1) * s1];
                float d = src[(y2 + 1) * s0 + (x2 + 1) * s1];
                dst[y * ow + x] = 0.25f * (a + b + c + d);
            }
        }

        Py_DECREF(dom);
        return (PyObject*)out;
    }

    npy_intp c = PyArray_DIM(dom, 2);
    npy_intp dims[3] = {oh, ow, c};
    PyArrayObject* out = (PyArrayObject*)PyArray_SimpleNew(3, dims, NPY_FLOAT32);
    if (!out) { Py_DECREF(dom); return NULL; }

    float* src = (float*)PyArray_DATA(dom);
    float* dst = (float*)PyArray_DATA(out);

    npy_intp s0 = PyArray_STRIDE(dom, 0) / (npy_intp)sizeof(float);
    npy_intp s1 = PyArray_STRIDE(dom, 1) / (npy_intp)sizeof(float);
    npy_intp s2 = PyArray_STRIDE(dom, 2) / (npy_intp)sizeof(float);

    for (npy_intp y = 0; y < oh; y++) {
        for (npy_intp x = 0; x < ow; x++) {
            npy_intp y2 = 2 * y;
            npy_intp x2 = 2 * x;
            float* outp = dst + (y * ow + x) * c;
            for (npy_intp ch = 0; ch < c; ch++) {
                float a = src[y2 * s0 + x2 * s1 + ch * s2];
                float b = src[(y2 + 1) * s0 + x2 * s1 + ch * s2];
                float cc = src[y2 * s0 + (x2 + 1) * s1 + ch * s2];
                float d = src[(y2 + 1) * s0 + (x2 + 1) * s1 + ch * s2];
                outp[ch] = 0.25f * (a + b + cc + d);
            }
        }
    }

    Py_DECREF(dom);
    return (PyObject*)out;
}

static PyObject* ff_linreg_error(PyObject* self, PyObject* args) {
    PyObject* d_obj;
    PyObject* r_obj;
    if (!PyArg_ParseTuple(args, "OO", &d_obj, &r_obj)) return NULL;

    PyArrayObject* d = (PyArrayObject*)PyArray_FROM_OTF(d_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* r = (PyArrayObject*)PyArray_FROM_OTF(r_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    if (!d || !r) {
        Py_XDECREF(d);
        Py_XDECREF(r);
        return NULL;
    }
    if (PyArray_SIZE(d) != PyArray_SIZE(r)) {
        PyErr_SetString(PyExc_ValueError, "linreg_error: d and r must have same size");
        Py_DECREF(d); Py_DECREF(r);
        return NULL;
    }

    npy_intp n = PyArray_SIZE(d);
    float* dp = (float*)PyArray_DATA(d);
    float* rp = (float*)PyArray_DATA(r);

    double sd = 0.0, sr = 0.0, sdd = 0.0, sdr = 0.0;
    for (npy_intp i = 0; i < n; i++) {
        double dv = (double)dp[i];
        double rv = (double)rp[i];
        sd += dv;
        sr += rv;
        sdd += dv * dv;
        sdr += dv * rv;
    }

    double N = (double)n;
    double denom = N * sdd - sd * sd;

    double s, o;
    if (denom == 0.0) {
        s = 0.0;
        o = sr / N;
    } else {
        s = (N * sdr - sd * sr) / denom;
        o = (sr - s * sd) / N;
    }

    double err = 0.0;
    for (npy_intp i = 0; i < n; i++) {
        double diff = s * (double)dp[i] + o - (double)rp[i];
        err += diff * diff;
    }
    err /= N;

    Py_DECREF(d);
    Py_DECREF(r);

    return Py_BuildValue("ddd", s, o, err);
}


static PyObject* ff_decode_iter_f32(PyObject* self, PyObject* args) {
    PyObject* cur0_obj;
    int h, w, c;
    PyObject* pool_blocks_obj;
    PyObject* pool_offsets_obj;
    PyObject* domain_yx_obj;
    PyObject* leaf_yx_obj;
    PyObject* leaf_pool_obj;
    PyObject* leaf_dom_obj;
    PyObject* leaf_tf_obj;
    int quantized;
    PyObject* codes_q_obj;
    PyObject* codes_f_obj;
    float s_clip, o_min, o_max;
    int iters;

    if (!PyArg_ParseTuple(
        args,
        "OiiiOOOOOOOiOOfffi:decode_iter_f32",
        &cur0_obj, &h, &w, &c,
        &pool_blocks_obj, &pool_offsets_obj, &domain_yx_obj,
        &leaf_yx_obj, &leaf_pool_obj, &leaf_dom_obj, &leaf_tf_obj,
        &quantized, &codes_q_obj, &codes_f_obj,
        &s_clip, &o_min, &o_max, &iters
    )) {
    return NULL;
}


    PyArrayObject* cur0 = (PyArrayObject*)PyArray_FROM_OTF(cur0_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    if (!cur0) return NULL;

    PyArrayObject* pool_blocks = (PyArrayObject*)PyArray_FROM_OTF(pool_blocks_obj, NPY_UINT16, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* pool_offsets = (PyArrayObject*)PyArray_FROM_OTF(pool_offsets_obj, NPY_UINT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* domain_yx = (PyArrayObject*)PyArray_FROM_OTF(domain_yx_obj, NPY_UINT16, NPY_ARRAY_IN_ARRAY);

    PyArrayObject* leaf_yx = (PyArrayObject*)PyArray_FROM_OTF(leaf_yx_obj, NPY_UINT16, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* leaf_pool = (PyArrayObject*)PyArray_FROM_OTF(leaf_pool_obj, NPY_UINT8, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* leaf_dom = (PyArrayObject*)PyArray_FROM_OTF(leaf_dom_obj, NPY_UINT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* leaf_tf = (PyArrayObject*)PyArray_FROM_OTF(leaf_tf_obj, NPY_UINT8, NPY_ARRAY_IN_ARRAY);

    if (!pool_blocks || !pool_offsets || !domain_yx || !leaf_yx || !leaf_pool || !leaf_dom || !leaf_tf) {
        Py_XDECREF(cur0);
        Py_XDECREF(pool_blocks); Py_XDECREF(pool_offsets); Py_XDECREF(domain_yx);
        Py_XDECREF(leaf_yx); Py_XDECREF(leaf_pool); Py_XDECREF(leaf_dom); Py_XDECREF(leaf_tf);
        return NULL;
    }

    PyArrayObject* codes_q = NULL;
    PyArrayObject* codes_f = NULL;

    if (quantized) {
        if (codes_q_obj != Py_None) {
            codes_q = (PyArrayObject*)PyArray_FROM_OTF(codes_q_obj, NPY_UINT8, NPY_ARRAY_IN_ARRAY);
        }
        if (!codes_q) {
            PyErr_SetString(PyExc_ValueError, "quantized decode requires codes_q");
            goto fail;
        }
    } else {
        if (codes_f_obj != Py_None) {
            codes_f = (PyArrayObject*)PyArray_FROM_OTF(codes_f_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
        }
        if (!codes_f) {
            PyErr_SetString(PyExc_ValueError, "float decode requires codes_f");
            goto fail;
        }
    }

    if (iters <= 0) {
        PyErr_SetString(PyExc_ValueError, "iters must be > 0");
        goto fail;
    }

    npy_intp out_nd = (c == 1) ? 2 : 3;
    npy_intp dims3[3] = {h, w, c};
    npy_intp dims2[2] = {h, w};

    PyArrayObject* cur = (PyArrayObject*)PyArray_SimpleNew(out_nd, (c == 1) ? dims2 : dims3, NPY_FLOAT32);
    PyArrayObject* nxt = (PyArrayObject*)PyArray_SimpleNew(out_nd, (c == 1) ? dims2 : dims3, NPY_FLOAT32);
    if (!cur || !nxt) {
        Py_XDECREF(cur); Py_XDECREF(nxt);
        PyErr_NoMemory();
        goto fail;
    }

    memcpy(PyArray_DATA(cur), PyArray_DATA(cur0), (size_t)PyArray_NBYTES(cur0));
    memset(PyArray_DATA(nxt), 0, (size_t)PyArray_NBYTES(nxt));

    uint16_t* pb = (uint16_t*)PyArray_DATA(pool_blocks);
    uint32_t* po = (uint32_t*)PyArray_DATA(pool_offsets);
    uint16_t* dyx = (uint16_t*)PyArray_DATA(domain_yx);

    uint16_t* lyx = (uint16_t*)PyArray_DATA(leaf_yx);
    uint8_t* lp = (uint8_t*)PyArray_DATA(leaf_pool);
    uint32_t* ld = (uint32_t*)PyArray_DATA(leaf_dom);
    uint8_t* lt = (uint8_t*)PyArray_DATA(leaf_tf);

    npy_intp n_leaves = PyArray_DIM(leaf_yx, 0);

    for (int it = 0; it < iters; it++) {
        float* curp = (float*)PyArray_DATA(cur);
        float* nxtp = (float*)PyArray_DATA(nxt);

        memset(nxtp, 0, (size_t)PyArray_NBYTES(nxt));

        for (npy_intp i = 0; i < n_leaves; i++) {
            uint16_t y = lyx[2*i + 0];
            uint16_t x = lyx[2*i + 1];

            uint8_t pooli = lp[i];
            uint16_t b = pb[pooli];

            uint32_t domi = ld[i];
            uint32_t base = po[pooli] + domi;
            uint16_t dy = dyx[2*base + 0];
            uint16_t dx = dyx[2*base + 1];

            uint8_t t = lt[i];

            for (int ch = 0; ch < c; ch++) {
                float s, o;
                if (quantized) {
                    uint8_t* cq = (uint8_t*)PyArray_DATA(codes_q);
                    uint8_t qs = cq[(i*c + ch)*2 + 0];
                    uint8_t qo = cq[(i*c + ch)*2 + 1];
                    s = (float)((double)qs * (2.0 * (double)s_clip) / 255.0 - (double)s_clip);
                    o = (float)((double)o_min + (double)qo * ((double)o_max - (double)o_min) / 255.0);
                } else {
                    float* cf = (float*)PyArray_DATA(codes_f);
                    s = cf[(i*c + ch)*2 + 0];
                    o = cf[(i*c + ch)*2 + 1];
                }

                for (uint16_t yy = 0; yy < b; yy++) {
                    for (uint16_t xx = 0; xx < b; xx++) {
                        uint16_t sy, sx;
                        switch (t) {
                            case 0: sy = yy; sx = xx; break;
                            case 1: sy = xx; sx = (uint16_t)(b - 1 - yy); break;
                            case 2: sy = (uint16_t)(b - 1 - yy); sx = (uint16_t)(b - 1 - xx); break;
                            case 3: sy = (uint16_t)(b - 1 - xx); sx = yy; break;
                            case 4: sy = yy; sx = (uint16_t)(b - 1 - xx); break;
                            case 5: sy = xx; sx = yy; break;
                            case 6: sy = (uint16_t)(b - 1 - yy); sx = xx; break;
                            case 7: sy = (uint16_t)(b - 1 - xx); sx = (uint16_t)(b - 1 - yy); break;
                            default: sy = yy; sx = xx; break;
                        }

                        uint16_t iy = (uint16_t)(dy + 2*sy);
                        uint16_t ix = (uint16_t)(dx + 2*sx);

                        float a, bb, cc, dd;
                        if (c == 1) {
                            a  = curp[iy * w + ix];
                            bb = curp[(iy + 1) * w + ix];
                            cc = curp[iy * w + (ix + 1)];
                            dd = curp[(iy + 1) * w + (ix + 1)];
                        } else {
                            npy_intp base0 = ((npy_intp)iy * w + (npy_intp)ix) * c + ch;
                            npy_intp base1 = ((npy_intp)(iy + 1) * w + (npy_intp)ix) * c + ch;
                            npy_intp base2 = ((npy_intp)iy * w + (npy_intp)(ix + 1)) * c + ch;
                            npy_intp base3 = ((npy_intp)(iy + 1) * w + (npy_intp)(ix + 1)) * c + ch;
                            a  = curp[base0];
                            bb = curp[base1];
                            cc = curp[base2];
                            dd = curp[base3];
                        }

                        float domv = 0.25f * (a + bb + cc + dd);
                        float outv = s * domv + o;
                        if (outv < 0.0f) outv = 0.0f;
                        if (outv > 1.0f) outv = 1.0f;

                        if (c == 1) {
                            nxtp[(y + yy) * w + (x + xx)] = outv;
                        } else {
                            nxtp[((npy_intp)(y + yy) * w + (npy_intp)(x + xx)) * c + ch] = outv;
                        }
                    }
                }
            }
        }

        PyArrayObject* tmp = cur; cur = nxt; nxt = tmp;
    }

    Py_DECREF(nxt);
    Py_DECREF(cur0);
    Py_DECREF(pool_blocks); Py_DECREF(pool_offsets); Py_DECREF(domain_yx);
    Py_DECREF(leaf_yx); Py_DECREF(leaf_pool); Py_DECREF(leaf_dom); Py_DECREF(leaf_tf);
    Py_XDECREF(codes_q);
    Py_XDECREF(codes_f);

    return (PyObject*)cur;

fail:
    Py_XDECREF(codes_q);
    Py_XDECREF(codes_f);
    Py_XDECREF(cur0);
    Py_XDECREF(pool_blocks); Py_XDECREF(pool_offsets); Py_XDECREF(domain_yx);
    Py_XDECREF(leaf_yx); Py_XDECREF(leaf_pool); Py_XDECREF(leaf_dom); Py_XDECREF(leaf_tf);
    return NULL;
}



static PyMethodDef Methods[] = {
    {"domains_yx", ff_domains_yx, METH_VARARGS, "Compute valid domain top-left coordinates (uint16 Nx2)."},
    {"ranges_yx", ff_ranges_yx, METH_VARARGS, "Compute valid range top-left coordinates (uint16 Nx2)."},
    {"extract_range", ff_extract_range, METH_VARARGS, "Extract bxb block (2D or 3D float32)."},
    {"extract_range_flat", ff_extract_range_flat, METH_VARARGS, "Extract bxb block flattened (2D float32)."},
    {"topk_dot", ff_topk_dot, METH_VARARGS, "Top-k dot products indices."},
    {"topk_from_subset", ff_topk_from_subset, METH_VARARGS, "Top-k dot over subset indices."},
    {"downsample2x2", ff_downsample2x2, METH_VARARGS, "2x2 average downsample (2D/3D float32)."},
    {"linreg_error", ff_linreg_error, METH_VARARGS, "Fit r ~ s*d + o, return (s,o,mse)."},
    {"decode_iter_f32", ff_decode_iter_f32, METH_VARARGS, "Decoder iteration (float32)."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef Module = {
    PyModuleDef_HEAD_INIT,
    "_cext",
    "fastfractal C extensions",
    -1,
    Methods
};

PyMODINIT_FUNC PyInit__cext(void) {
    PyObject* m = PyModule_Create(&Module);
    if (!m) return NULL;
    import_array();
    return m;
}
