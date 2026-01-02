#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL FASTFRACTAL_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#include <math.h>
#include <stdint.h>

static inline double dot_f32(const float* a, const float* b, npy_intp n) {
    double s = 0.0;
    for (npy_intp i = 0; i < n; i++) s += (double)a[i] * (double)b[i];
    return s;
}

static inline double sumsq_f32(const float* a, npy_intp n) {
    double s = 0.0;
    for (npy_intp i = 0; i < n; i++) {
        double v = (double)a[i];
        s += v * v;
    }
    return s;
}

static inline double sum_f32(const float* a, npy_intp n) {
    double s = 0.0;
    for (npy_intp i = 0; i < n; i++) s += (double)a[i];
    return s;
}

static inline double clipd(double x, double lo, double hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

static inline int clamp_i32(int v, int lo, int hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static inline int quant_s(double s, double s_clip) {
    double sc = clipd(s, -s_clip, s_clip);
    double qf = (sc + s_clip) * 255.0 / (2.0 * s_clip);
    int q = (int)llround(qf);
    return clamp_i32(q, 0, 255);
}

static inline double dequant_s(int q, double s_clip) {
    return (double)q * (2.0 * s_clip) / 255.0 - s_clip;
}

static inline int quant_o(double o, double o_min, double o_max) {
    double oc = clipd(o, o_min, o_max);
    double den = (o_max - o_min);
    double qf = (oc - o_min) * 255.0 / den;
    int q = (int)llround(qf);
    return clamp_i32(q, 0, 255);
}

static inline double dequant_o(int q, double o_min, double o_max) {
    return o_min + (double)q * (o_max - o_min) / 255.0;
}

static PyObject* encode_leaf_best(PyObject* self, PyObject* args) {
    PyObject *img_obj=NULL, *tf_flat_obj=NULL, *tf_sum_obj=NULL, *tf_sum2_obj=NULL;
    PyObject *map_dom_obj=NULL, *map_tf_obj=NULL, *cand_obj=NULL;

    PyObject* sset_obj = Py_None;

    int y=0, x=0, block=0;
    double s_clip=0.0, o_min=0.0, o_max=0.0;
    int quantized=0;

    if (!PyArg_ParseTuple(
            args,
            "OOOOOOiiiOdddi|O",
            &img_obj, &tf_flat_obj, &tf_sum_obj, &tf_sum2_obj,
            &map_dom_obj, &map_tf_obj,
            &y, &x, &block,
            &cand_obj,
            &s_clip, &o_min, &o_max,
            &quantized,
            &sset_obj
        )) {
        return NULL;
    }

    if (block <= 0) { PyErr_SetString(PyExc_ValueError, "block must be positive"); return NULL; }
    if (!(s_clip > 0.0)) { PyErr_SetString(PyExc_ValueError, "s_clip must be > 0"); return NULL; }
    if (!(o_max > o_min)) { PyErr_SetString(PyExc_ValueError, "o_max must be > o_min"); return NULL; }

    PyArrayObject* img = (PyArrayObject*)PyArray_FROM_OTF(img_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    if (!img) return NULL;

    PyArrayObject* tf_flat = (PyArrayObject*)PyArray_FROM_OTF(tf_flat_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    if (!tf_flat) { Py_DECREF(img); return NULL; }

    PyArrayObject* map_dom = (PyArrayObject*)PyArray_FROM_OTF(map_dom_obj, NPY_UINT32, NPY_ARRAY_IN_ARRAY);
    if (!map_dom) { Py_DECREF(img); Py_DECREF(tf_flat); return NULL; }

    PyArrayObject* map_tf = (PyArrayObject*)PyArray_FROM_OTF(map_tf_obj, NPY_UINT8, NPY_ARRAY_IN_ARRAY);
    if (!map_tf) { Py_DECREF(img); Py_DECREF(tf_flat); Py_DECREF(map_dom); return NULL; }

    PyArrayObject* cand = NULL;
    int cand_is_i32 = 0;
    cand = (PyArrayObject*)PyArray_FROM_OTF(cand_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    if (cand) {
        cand_is_i32 = 1;
    } else {
        PyErr_Clear();
        cand = (PyArrayObject*)PyArray_FROM_OTF(cand_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
        if (!cand) {
            Py_DECREF(img); Py_DECREF(tf_flat); Py_DECREF(map_dom); Py_DECREF(map_tf);
            return NULL;
        }
        cand_is_i32 = 0;
    }

    PyArrayObject* tf_sum = NULL;
    PyArrayObject* tf_sum2 = NULL;

    if (tf_sum_obj != Py_None) {
        tf_sum = (PyArrayObject*)PyArray_FROM_OTF(tf_sum_obj, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
        if (!tf_sum) goto fail;
    }
    if (tf_sum2_obj != Py_None) {
        tf_sum2 = (PyArrayObject*)PyArray_FROM_OTF(tf_sum2_obj, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
        if (!tf_sum2) goto fail;
    }

    if (PyArray_NDIM(tf_flat) != 3) {
        PyErr_SetString(PyExc_ValueError, "tf_flat must be 3D (n_entries, C, n_pix)");
        goto fail;
    }

    npy_intp n_entries = PyArray_DIM(tf_flat, 0);
    npy_intp C = PyArray_DIM(tf_flat, 1);
    npy_intp n_pix = PyArray_DIM(tf_flat, 2);
    npy_intp expect_pix = (npy_intp)block * (npy_intp)block;
    if (n_pix != expect_pix) {
        PyErr_SetString(PyExc_ValueError, "tf_flat.shape[2] must equal block*block");
        goto fail;
    }
    if (!(C == 1 || C == 3)) {
        PyErr_SetString(PyExc_ValueError, "only C==1 or C==3 supported");
        goto fail;
    }

    if (PyArray_NDIM(map_dom) != 1 || PyArray_DIM(map_dom, 0) != n_entries) {
        PyErr_SetString(PyExc_ValueError, "map_dom must be 1D of length tf_flat.shape[0]");
        goto fail;
    }
    if (PyArray_NDIM(map_tf) != 1 || PyArray_DIM(map_tf, 0) != n_entries) {
        PyErr_SetString(PyExc_ValueError, "map_tf must be 1D of length tf_flat.shape[0]");
        goto fail;
    }

    if (tf_sum && (PyArray_NDIM(tf_sum) != 2 || PyArray_DIM(tf_sum, 0) != n_entries || PyArray_DIM(tf_sum, 1) != C)) {
        PyErr_SetString(PyExc_ValueError, "tf_sum must be float64 (n_entries, C)");
        goto fail;
    }
    if (tf_sum2 && (PyArray_NDIM(tf_sum2) != 2 || PyArray_DIM(tf_sum2, 0) != n_entries || PyArray_DIM(tf_sum2, 1) != C)) {
        PyErr_SetString(PyExc_ValueError, "tf_sum2 must be float64 (n_entries, C)");
        goto fail;
    }

    if (PyArray_NDIM(cand) != 1) {
        PyErr_SetString(PyExc_ValueError, "cand must be 1D");
        goto fail;
    }
    npy_intp m = PyArray_DIM(cand, 0);
    if (m <= 0) {
        npy_intp cdims[2] = {C, 2};
        PyArrayObject* codes = (PyArrayObject*)PyArray_SimpleNew(2, cdims, quantized ? NPY_UINT8 : NPY_FLOAT32);
        if (!codes) goto fail;

        PyObject* tup = PyTuple_New(4);
        if (!tup) { Py_DECREF(codes); goto fail; }
        PyTuple_SET_ITEM(tup, 0, PyLong_FromLong(0));
        PyTuple_SET_ITEM(tup, 1, PyLong_FromLong(0));
        PyTuple_SET_ITEM(tup, 2, (PyObject*)codes);
        PyTuple_SET_ITEM(tup, 3, PyFloat_FromDouble(INFINITY));

        Py_DECREF(img); Py_DECREF(tf_flat); Py_DECREF(map_dom); Py_DECREF(map_tf); Py_DECREF(cand);
        Py_XDECREF(tf_sum); Py_XDECREF(tf_sum2);
        return tup;
    }

    int img_nd = PyArray_NDIM(img);
    if (!(img_nd == 2 || img_nd == 3)) {
        PyErr_SetString(PyExc_ValueError, "img must be 2D or 3D float32");
        goto fail;
    }
    if (img_nd == 2) {
        if (C != 1) { PyErr_SetString(PyExc_ValueError, "tf_flat has C!=1 but img is 2D"); goto fail; }
        npy_intp H = PyArray_DIM(img, 0), W = PyArray_DIM(img, 1);
        if (y < 0 || x < 0 || (npy_intp)y + block > H || (npy_intp)x + block > W) {
            PyErr_SetString(PyExc_ValueError, "range block out of bounds");
            goto fail;
        }
    } else {
        npy_intp H = PyArray_DIM(img, 0), W = PyArray_DIM(img, 1), imgC = PyArray_DIM(img, 2);
        if (imgC != C) { PyErr_SetString(PyExc_ValueError, "img.shape[2] must equal tf_flat.shape[1]"); goto fail; }
        if (y < 0 || x < 0 || (npy_intp)y + block > H || (npy_intp)x + block > W) {
            PyErr_SetString(PyExc_ValueError, "range block out of bounds");
            goto fail;
        }
    }

    float* rbuf = (float*)PyMem_Malloc((size_t)(C * n_pix) * sizeof(float));
    if (!rbuf) { PyErr_NoMemory(); goto fail; }

    double sumR[3]  = {0.0, 0.0, 0.0};
    double sumRR[3] = {0.0, 0.0, 0.0};

    char* imgp = (char*)PyArray_DATA(img);
    npy_intp s0 = PyArray_STRIDE(img, 0);
    npy_intp s1 = PyArray_STRIDE(img, 1);
    npy_intp s2 = (img_nd == 3) ? PyArray_STRIDE(img, 2) : 0;

    npy_intp p = 0;
    for (int yy = 0; yy < block; yy++) {
        for (int xx = 0; xx < block; xx++, p++) {
            if (img_nd == 2) {
                float v = *(float*)(imgp + (npy_intp)(y + yy)*s0 + (npy_intp)(x + xx)*s1);
                rbuf[p] = v;
                double dv = (double)v;
                sumR[0] += dv;
                sumRR[0] += dv*dv;
            } else {
                for (npy_intp ch = 0; ch < C; ch++) {
                    float v = *(float*)(imgp + (npy_intp)(y + yy)*s0 + (npy_intp)(x + xx)*s1 + ch*s2);
                    rbuf[ch*n_pix + p] = v;
                    double dv = (double)v;
                    sumR[ch] += dv;
                    sumRR[ch] += dv*dv;
                }
            }
        }
    }

    float* tfp = (float*)PyArray_DATA(tf_flat);
    npy_intp tf_s0 = PyArray_STRIDE(tf_flat, 0) / (npy_intp)sizeof(float);
    npy_intp tf_s1 = PyArray_STRIDE(tf_flat, 1) / (npy_intp)sizeof(float);

    uint32_t* map_dom_p = (uint32_t*)PyArray_DATA(map_dom);
    uint8_t*  map_tf_p  = (uint8_t*)PyArray_DATA(map_tf);

    double* tf_sum_p  = tf_sum  ? (double*)PyArray_DATA(tf_sum)  : NULL;
    double* tf_sum2_p = tf_sum2 ? (double*)PyArray_DATA(tf_sum2) : NULL;
    npy_intp sum_s0   = tf_sum  ? (PyArray_STRIDE(tf_sum, 0) / (npy_intp)sizeof(double)) : 0;
    npy_intp sum_s1   = tf_sum  ? (PyArray_STRIDE(tf_sum, 1) / (npy_intp)sizeof(double)) : 0;
    npy_intp sum2_s0  = tf_sum2 ? (PyArray_STRIDE(tf_sum2, 0) / (npy_intp)sizeof(double)) : 0;
    npy_intp sum2_s1  = tf_sum2 ? (PyArray_STRIDE(tf_sum2, 1) / (npy_intp)sizeof(double)) : 0;

    const double dn = (double)n_pix;
    const double inv_n = 1.0 / dn;

    double best_mse = INFINITY;
    npy_intp best_k = 0;

    double best_s_f[3] = {0,0,0};
    double best_o_f[3] = {0,0,0};
    uint8_t best_s_q[3] = {0,0,0};
    uint8_t best_o_q[3] = {0,0,0};

    const int32_t* cand_i32 = cand_is_i32 ? (const int32_t*)PyArray_DATA(cand) : NULL;
    const int64_t* cand_i64 = cand_is_i32 ? NULL : (const int64_t*)PyArray_DATA(cand);

    Py_BEGIN_ALLOW_THREADS

    for (npy_intp ii = 0; ii < m; ii++) {
        int64_t kk64 = cand_is_i32 ? (int64_t)cand_i32[ii] : cand_i64[ii];
        if (kk64 < 0 || kk64 >= (int64_t)n_entries) continue;
        npy_intp k = (npy_intp)kk64;

        if (C == 1) {
            const float* dom = tfp + k*tf_s0 + 0*tf_s1;

            double sumD  = tf_sum_p  ? *(tf_sum_p  + k*sum_s0  + 0*sum_s1)  : sum_f32(dom, n_pix);
            double sumDD = tf_sum2_p ? *(tf_sum2_p + k*sum2_s0 + 0*sum2_s1) : sumsq_f32(dom, n_pix);
            double sumRD = dot_f32(dom, rbuf, n_pix);

            double denom = dn * sumDD - sumD * sumD;
            double s0v, o0v;
            if (fabs(denom) < 1e-18) {
                s0v = 0.0;
                o0v = sumR[0] * inv_n;
            } else {
                s0v = (dn * sumRD - sumD * sumR[0]) / denom;
                o0v = (sumR[0] - s0v * sumD) * inv_n;
            }

            double s1v = clipd(s0v, -s_clip, s_clip);
            double o1v = clipd(o0v, o_min, o_max);

            if (quantized) {
                int qs = quant_s(s1v, s_clip);
                int qo = quant_o(o1v, o_min, o_max);
                double s2 = dequant_s(qs, s_clip);
                double o2 = dequant_o(qo, o_min, o_max);

                double sse = sumRR[0]
                    + (s2*s2)*sumDD
                    + dn*(o2*o2)
                    - 2.0*s2*sumRD
                    - 2.0*o2*sumR[0]
                    + 2.0*s2*o2*sumD;

                double mse = sse * inv_n;
                if (mse < best_mse) {
                    best_mse = mse;
                    best_k = k;
                    best_s_q[0] = (uint8_t)qs;
                    best_o_q[0] = (uint8_t)qo;
                }
            } else {
                double sse = sumRR[0]
                    + (s1v*s1v)*sumDD
                    + dn*(o1v*o1v)
                    - 2.0*s1v*sumRD
                    - 2.0*o1v*sumR[0]
                    + 2.0*s1v*o1v*sumD;

                double mse = sse * inv_n;
                if (mse < best_mse) {
                    best_mse = mse;
                    best_k = k;
                    best_s_f[0] = s1v;
                    best_o_f[0] = o1v;
                }
            }
        } else {
            double sse_sum = 0.0;

            double s1v[3] = {0,0,0};
            double o1v[3] = {0,0,0};
            uint8_t qs_v[3] = {0,0,0};
            uint8_t qo_v[3] = {0,0,0};

            for (npy_intp ch = 0; ch < C; ch++) {
                const float* dom = tfp + k*tf_s0 + ch*tf_s1;
                const float* rr  = rbuf + ch*n_pix;

                double sumD  = tf_sum_p  ? *(tf_sum_p  + k*sum_s0  + ch*sum_s1)  : sum_f32(dom, n_pix);
                double sumDD = tf_sum2_p ? *(tf_sum2_p + k*sum2_s0 + ch*sum2_s1) : sumsq_f32(dom, n_pix);
                double sumRD = dot_f32(dom, rr, n_pix);

                double denom = dn * sumDD - sumD * sumD;
                double s0c, o0c;
                if (fabs(denom) < 1e-18) {
                    s0c = 0.0;
                    o0c = sumR[ch] * inv_n;
                } else {
                    s0c = (dn * sumRD - sumD * sumR[ch]) / denom;
                    o0c = (sumR[ch] - s0c * sumD) * inv_n;
                }

                double s1c = clipd(s0c, -s_clip, s_clip);
                double o1c = clipd(o0c, o_min, o_max);

                if (quantized) {
                    int qs = quant_s(s1c, s_clip);
                    int qo = quant_o(o1c, o_min, o_max);
                    double s2 = dequant_s(qs, s_clip);
                    double o2 = dequant_o(qo, o_min, o_max);

                    double sse = sumRR[ch]
                        + (s2*s2)*sumDD
                        + dn*(o2*o2)
                        - 2.0*s2*sumRD
                        - 2.0*o2*sumR[ch]
                        + 2.0*s2*o2*sumD;

                    sse_sum += sse;
                    qs_v[ch] = (uint8_t)qs;
                    qo_v[ch] = (uint8_t)qo;
                } else {
                    double sse = sumRR[ch]
                        + (s1c*s1c)*sumDD
                        + dn*(o1c*o1c)
                        - 2.0*s1c*sumRD
                        - 2.0*o1c*sumR[ch]
                        + 2.0*s1c*o1c*sumD;

                    sse_sum += sse;
                    s1v[ch] = s1c;
                    o1v[ch] = o1c;
                }
            }

            double mse = sse_sum / ((double)C * dn);
            if (mse < best_mse) {
                best_mse = mse;
                best_k = k;
                if (quantized) {
                    for (npy_intp ch = 0; ch < C; ch++) {
                        best_s_q[ch] = qs_v[ch];
                        best_o_q[ch] = qo_v[ch];
                    }
                } else {
                    for (npy_intp ch = 0; ch < C; ch++) {
                        best_s_f[ch] = s1v[ch];
                        best_o_f[ch] = o1v[ch];
                    }
                }
            }
        }
    }

    Py_END_ALLOW_THREADS

    npy_intp cdims[2] = {C, 2};
    PyArrayObject* codes = (PyArrayObject*)PyArray_SimpleNew(2, cdims, quantized ? NPY_UINT8 : NPY_FLOAT32);
    if (!codes) { PyMem_Free(rbuf); goto fail; }

    if (quantized) {
        uint8_t* outp = (uint8_t*)PyArray_DATA(codes);
        for (npy_intp ch = 0; ch < C; ch++) {
            outp[ch*2 + 0] = best_s_q[ch];
            outp[ch*2 + 1] = best_o_q[ch];
        }
    } else {
        float* outp = (float*)PyArray_DATA(codes);
        for (npy_intp ch = 0; ch < C; ch++) {
            outp[ch*2 + 0] = (float)best_s_f[ch];
            outp[ch*2 + 1] = (float)best_o_f[ch];
        }
    }

    int best_dom = (int)map_dom_p[best_k];
    int best_tf  = (int)map_tf_p[best_k];

    PyMem_Free(rbuf);

    PyObject* tup = PyTuple_New(4);
    if (!tup) { Py_DECREF(codes); goto fail; }
    PyTuple_SET_ITEM(tup, 0, PyLong_FromLong(best_dom));
    PyTuple_SET_ITEM(tup, 1, PyLong_FromLong(best_tf));
    PyTuple_SET_ITEM(tup, 2, (PyObject*)codes);
    PyTuple_SET_ITEM(tup, 3, PyFloat_FromDouble(best_mse));

    Py_DECREF(img);
    Py_DECREF(tf_flat);
    Py_DECREF(map_dom);
    Py_DECREF(map_tf);
    Py_DECREF(cand);
    Py_XDECREF(tf_sum);
    Py_XDECREF(tf_sum2);
    return tup;

fail:
    Py_DECREF(img);
    Py_DECREF(tf_flat);
    Py_DECREF(map_dom);
    Py_DECREF(map_tf);
    Py_DECREF(cand);
    Py_XDECREF(tf_sum);
    Py_XDECREF(tf_sum2);
    return NULL;
}

PyObject* fastfractal_encode_leaf_best(PyObject* self, PyObject* args) {
    return encode_leaf_best(self, args);
}
