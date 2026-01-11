#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL FASTFRACTAL_ARRAY_API
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdint.h>

enum {
    REG_LINEAR = 0,
    REG_QUADREG = 1,
    REG_SIGMOID = 2,
    REG_FAST = 3,
    REG_HUBER = 4,
    REG_CAUCHY = 5
};

static int streq_ci(const char* a, const char* b) {
    while (*a && *b) {
        char ca = *a, cb = *b;
        if (ca >= 'A' && ca <= 'Z') ca = (char)(ca - 'A' + 'a');
        if (cb >= 'A' && cb <= 'Z') cb = (char)(cb - 'A' + 'a');
        if (ca != cb) return 0;
        a++; b++;
    }
    return (*a == '\0' && *b == '\0');
}

static int parse_regression_selector(PyObject* obj, int* out_reg) {
    *out_reg = REG_LINEAR;
    if (!obj || obj == Py_None) return 1;

    if (PyLong_Check(obj)) {
        long v = PyLong_AsLong(obj);
        if (PyErr_Occurred()) return 0;
        if (v < 0 || v > 5) {
            PyErr_SetString(PyExc_ValueError, "regression id out of range (expected 0..5)");
            return 0;
        }
        *out_reg = (int)v;
        return 1;
    }

    if (PyUnicode_Check(obj)) {
        const char* s = PyUnicode_AsUTF8(obj);
        if (!s) return 0;

        if (streq_ci(s, "linear") || streq_ci(s, "lin") || streq_ci(s, "least_squares")) {
            *out_reg = REG_LINEAR; return 1;
        }
        if (streq_ci(s, "quadreg") || streq_ci(s, "quadratic") || streq_ci(s, "ridge")) {
            *out_reg = REG_QUADREG; return 1;
        }
        if (streq_ci(s, "sigmoid") || streq_ci(s, "sigmoidal") || streq_ci(s, "logistic")) {
            *out_reg = REG_SIGMOID; return 1;
        }
        if (streq_ci(s, "fast") || streq_ci(s, "mean")) {
            *out_reg = REG_FAST; return 1;
        }
        if (streq_ci(s, "huber")) {
            *out_reg = REG_HUBER; return 1;
        }
        if (streq_ci(s, "cauchy")) {
            *out_reg = REG_CAUCHY; return 1;
        }

        PyErr_Format(PyExc_ValueError, "unknown regression name: %s", s);
        return 0;
    }

    PyErr_SetString(PyExc_TypeError, "regression selector must be int or str");
    return 0;
}

static inline double dot_f32(const float* a, const float* b, npy_intp n) {
    double s=0.0;
    for(npy_intp i=0;i<n;i++) s += (double)a[i]*(double)b[i];
    return s;
}
static inline double sum_f32(const float* a, npy_intp n) {
    double s=0.0;
    for(npy_intp i=0;i<n;i++) s += (double)a[i];
    return s;
}
static inline double sumsq_f32(const float* a, npy_intp n) {
    double s=0.0;
    for(npy_intp i=0;i<n;i++){double v=(double)a[i]; s+=v*v;}
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

static inline void weighted_solve(
    double sumW,double sumWD,double sumWDD,double sumWR,double sumWDR,
    double* s_out,double* o_out
) {
    double denom = sumW*sumWDD - sumWD*sumWD;
    if (fabs(denom) < 1e-18 || sumW <= 1e-18) {
        *s_out = 0.0;
        *o_out = (sumW>1e-18 ? sumWR/sumW : 0.0);
        return;
    }
    double s = (sumW*sumWDR - sumWD*sumWR)/denom;
    double o = (sumWR - s*sumWD)/sumW;
    *s_out = s; *o_out = o;
}

static void solve_regression(
    int reg,
    const float* dom,const float* r,
    npy_intp n_pix,double dn,
    double sumD,double sumDD,double sumR,double sumRD,
    double* out_s,double* out_o
) {
    if(reg == REG_FAST) {
        *out_s = 0.0;
        *out_o = sumR/dn;
        return;
    }

    double denom = dn*sumDD - sumD*sumD;
    double s=0.0, o=0.0;
    if(fabs(denom) < 1e-18) {
        o = sumR/dn;
    } else {
        s = (dn*sumRD - sumD*sumR)/denom;
        o = (sumR - s*sumD)/dn;
    }

    if(reg == REG_LINEAR) {
        *out_s=s; *out_o=o;
        return;
    }

    if(reg == REG_HUBER || reg == REG_CAUCHY || reg == REG_SIGMOID) {
        const double huber_delta = 0.05;
        const double cauchy_scale2=0.05*0.05;
        const double sig_alpha=12.0, sig_beta=0.05;
        int iters=2;
        for(int it=0;it<iters;it++){
            double sumW=0,sumWD=0,sumWDD=0,sumWR=0,sumWDR=0;
            for(npy_intp i=0;i<n_pix;i++){
                double dv = dom[i], rv = r[i];
                double res=rv-(s*dv+o);
                double w=1.0;
                double absr=fabs(res);
                if(reg==REG_HUBER){
                    w = (absr<=huber_delta?1.0:(huber_delta/absr));
                } else if(reg == REG_CAUCHY){
                    w = 1.0/(1.0 + (res*res)/cauchy_scale2);
                } else {
                    w = 1.0/(1.0+exp(sig_alpha*(absr - sig_beta)));
                }
                sumW   += w;
                sumWD  += w*dv;
                sumWDD += w*dv*dv;
                sumWR  += w*rv;
                sumWDR += w*dv*rv;
            }
            weighted_solve(sumW,sumWD,sumWDD,sumWR,sumWDR,&s,&o);
        }
        *out_s=s; *out_o=o;
        return;
    }

    *out_s=s; *out_o=o;
}

typedef struct { double score; npy_int64 idx; } ScoreIdx;
static void heap_sift_down(ScoreIdx* heap, npy_intp n, npy_intp i) {
    while(1){
        npy_intp l=2*i+1, r=l+1, smallest=i;
        if(l<n && heap[l].score<heap[smallest].score) smallest=l;
        if(r<n && heap[r].score<heap[smallest].score) smallest=r;
        if(smallest==i) break;
        ScoreIdx tmp=heap[i]; heap[i]=heap[smallest]; heap[smallest]=tmp;
        i=smallest;
    }
}

static PyObject* topk_from_subset(PyObject* self, PyObject* args);

PyObject* fastfractal_encode_leaf_best(PyObject* self, PyObject* args);

static PyObject* encode_leaf_best_impl(PyObject* self, PyObject* args) {
    PyObject *img_obj, *tf_flat_obj, *tf_sum_obj, *tf_sum2_obj;
    PyObject *map_dom_obj, *map_tf_obj, *cand_obj;
    int y,x,block;
    double s_clip,o_min,o_max;
    int quantized;

    PyObject* opt1 = Py_None;
    PyObject* opt2 = Py_None;

    if(!PyArg_ParseTuple(args,
        "OOOOOOiiiOdddi|OO",
        &img_obj,&tf_flat_obj,&tf_sum_obj,&tf_sum2_obj,
        &map_dom_obj,&map_tf_obj,
        &y,&x,&block,
        &cand_obj,
        &s_clip,&o_min,&o_max,
        &quantized,
        &opt1,&opt2
    )) return NULL;

    if(block<=0){PyErr_SetString(PyExc_ValueError,"bad block");return NULL;}
    if(s_clip<=0){PyErr_SetString(PyExc_ValueError,"s_clip<=0");return NULL;}
    if(!(o_max>o_min)){PyErr_SetString(PyExc_ValueError,"o_max<=o_min");return NULL;}

    PyObject* reg_obj = Py_None;
    if(opt2!=Py_None) reg_obj=opt2;
    else if(opt1!=Py_None) {
        if(PyLong_Check(opt1)||PyUnicode_Check(opt1))
            reg_obj=opt1;
    }

    int reg_kind=REG_LINEAR;
    if(!parse_regression_selector(reg_obj,&reg_kind)) return NULL;

    PyArrayObject* img = (PyArrayObject*)
        PyArray_FROM_OTF(img_obj,NPY_FLOAT32,NPY_ARRAY_IN_ARRAY);
    if(!img) return NULL;

    PyArrayObject* tf_flat = (PyArrayObject*)
        PyArray_FROM_OTF(tf_flat_obj,NPY_FLOAT32,NPY_ARRAY_IN_ARRAY);
    if(!tf_flat){Py_DECREF(img);return NULL;}

    PyArrayObject* map_dom = (PyArrayObject*)
        PyArray_FROM_OTF(map_dom_obj,NPY_UINT32,NPY_ARRAY_IN_ARRAY);
    if(!map_dom){Py_DECREF(img);Py_DECREF(tf_flat);return NULL;}

    PyArrayObject* map_tf  = (PyArrayObject*)
        PyArray_FROM_OTF(map_tf_obj,NPY_UINT8,NPY_ARRAY_IN_ARRAY);
    if(!map_tf){Py_DECREF(img);Py_DECREF(tf_flat);Py_DECREF(map_dom);return NULL;}

    PyArrayObject* cand=NULL;
    int cand_is_i32=0;
    cand = (PyArrayObject*)
        PyArray_FROM_OTF(cand_obj,NPY_INT32,NPY_ARRAY_IN_ARRAY);
    if(cand){ cand_is_i32=1; }
    else{
        PyErr_Clear();
        cand = (PyArrayObject*)PyArray_FROM_OTF(cand_obj,NPY_INT64,NPY_ARRAY_IN_ARRAY);
        if(!cand){
            Py_DECREF(img);Py_DECREF(tf_flat);Py_DECREF(map_dom);Py_DECREF(map_tf);
            return NULL;
        }
        cand_is_i32=0;
    }

    PyArrayObject *tf_sum=NULL,*tf_sum2=NULL;
    if(tf_sum_obj!=Py_None){
        tf_sum = (PyArrayObject*)
            PyArray_FROM_OTF(tf_sum_obj,NPY_FLOAT64,NPY_ARRAY_IN_ARRAY);
        if(!tf_sum) {}
    }
    if(tf_sum2_obj!=Py_None){
        tf_sum2 = (PyArrayObject*)
            PyArray_FROM_OTF(tf_sum2_obj,NPY_FLOAT64,NPY_ARRAY_IN_ARRAY);
        if(!tf_sum2) {}
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
        npy_intp cdims0[2] = {C, 2};
        PyArrayObject* codes0 = (PyArrayObject*)PyArray_SimpleNew(2, cdims0, quantized ? NPY_UINT8 : NPY_FLOAT32);
        if (!codes0) goto fail;
        PyObject* tup0 = PyTuple_New(4);
        if (!tup0) { Py_DECREF(codes0); goto fail; }
        PyTuple_SET_ITEM(tup0, 0, PyLong_FromLong(0));
        PyTuple_SET_ITEM(tup0, 1, PyLong_FromLong(0));
        PyTuple_SET_ITEM(tup0, 2, (PyObject*)codes0);
        PyTuple_SET_ITEM(tup0, 3, PyFloat_FromDouble(INFINITY));
        Py_DECREF(img); Py_DECREF(tf_flat); Py_DECREF(map_dom); Py_DECREF(map_tf); Py_DECREF(cand);
        Py_XDECREF(tf_sum); Py_XDECREF(tf_sum2);
        return tup0;
    }

    int img_nd = PyArray_NDIM(img);
    if (!(img_nd == 2 || img_nd == 3)) {
        PyErr_SetString(PyExc_ValueError, "img must be 2D or 3D float32");
        goto fail;
    }

    npy_intp H = PyArray_DIM(img, 0);
    npy_intp W = PyArray_DIM(img, 1);

    if (y < 0 || x < 0 || (npy_intp)y + block > H || (npy_intp)x + block > W) {
        PyErr_SetString(PyExc_ValueError, "range block out of bounds");
        goto fail;
    }

    if (img_nd == 2) {
        if (C != 1) {
            PyErr_SetString(PyExc_ValueError, "tf_flat has C!=1 but img is 2D");
            goto fail;
        }
    } else {
        npy_intp imgC = PyArray_DIM(img, 2);
        if (imgC != C) {
            PyErr_SetString(PyExc_ValueError, "img.shape[2] must equal tf_flat.shape[1]");
            goto fail;
        }
    }

    float* rbuf = (float*)PyMem_Malloc((size_t)(C * n_pix) * sizeof(float));
    if (!rbuf) { PyErr_NoMemory(); goto fail; }

    double sumR[3]  = {0.0, 0.0, 0.0};
    double sumRR[3] = {0.0, 0.0, 0.0};

    char* imgp = (char*)PyArray_DATA(img);
    npy_intp is0 = PyArray_STRIDE(img, 0);
    npy_intp is1 = PyArray_STRIDE(img, 1);
    npy_intp is2 = (img_nd == 3) ? PyArray_STRIDE(img, 2) : 0;

    npy_intp p = 0;
    for (int yy = 0; yy < block; yy++) {
        for (int xx = 0; xx < block; xx++, p++) {
            if (img_nd == 2) {
                float v = *(float*)(imgp + (npy_intp)(y + yy)*is0 + (npy_intp)(x + xx)*is1);
                rbuf[p] = v;
                double dv = (double)v;
                sumR[0] += dv;
                sumRR[0] += dv*dv;
            } else {
                for (npy_intp ch = 0; ch < C; ch++) {
                    float v = *(float*)(imgp + (npy_intp)(y + yy)*is0 + (npy_intp)(x + xx)*is1 + ch*is2);
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

    npy_intp sum_s0  = tf_sum  ? (PyArray_STRIDE(tf_sum, 0)  / (npy_intp)sizeof(double)) : 0;
    npy_intp sum_s1  = tf_sum  ? (PyArray_STRIDE(tf_sum, 1)  / (npy_intp)sizeof(double)) : 0;
    npy_intp sum2_s0 = tf_sum2 ? (PyArray_STRIDE(tf_sum2, 0) / (npy_intp)sizeof(double)) : 0;
    npy_intp sum2_s1 = tf_sum2 ? (PyArray_STRIDE(tf_sum2, 1) / (npy_intp)sizeof(double)) : 0;

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

            double s0v, o0v;
            solve_regression(reg_kind, dom, rbuf, n_pix, dn, sumD, sumDD, sumR[0], sumRD, &s0v, &o0v);

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

            double s1v_arr[3] = {0,0,0};
            double o1v_arr[3] = {0,0,0};
            uint8_t qs_arr[3] = {0,0,0};
            uint8_t qo_arr[3] = {0,0,0};

            for (npy_intp ch = 0; ch < C; ch++) {
                const float* dom = tfp + k*tf_s0 + ch*tf_s1;
                const float* rr  = rbuf + ch*n_pix;

                double sumD  = tf_sum_p  ? *(tf_sum_p  + k*sum_s0  + ch*sum_s1)  : sum_f32(dom, n_pix);
                double sumDD = tf_sum2_p ? *(tf_sum2_p + k*sum2_s0 + ch*sum2_s1) : sumsq_f32(dom, n_pix);
                double sumRD = dot_f32(dom, rr, n_pix);

                double s0v, o0v;
                solve_regression(reg_kind, dom, rr, n_pix, dn, sumD, sumDD, sumR[ch], sumRD, &s0v, &o0v);

                double s1v = clipd(s0v, -s_clip, s_clip);
                double o1v = clipd(o0v, o_min, o_max);

                if (quantized) {
                    int qs = quant_s(s1v, s_clip);
                    int qo = quant_o(o1v, o_min, o_max);
                    double s2 = dequant_s(qs, s_clip);
                    double o2 = dequant_o(qo, o_min, o_max);

                    double sse = sumRR[ch]
                        + (s2*s2)*sumDD
                        + dn*(o2*o2)
                        - 2.0*s2*sumRD
                        - 2.0*o2*sumR[ch]
                        + 2.0*s2*o2*sumD;

                    sse_sum += sse;
                    qs_arr[ch] = (uint8_t)qs;
                    qo_arr[ch] = (uint8_t)qo;
                } else {
                    double sse = sumRR[ch]
                        + (s1v*s1v)*sumDD
                        + dn*(o1v*o1v)
                        - 2.0*s1v*sumRD
                        - 2.0*o1v*sumR[ch]
                        + 2.0*s1v*o1v*sumD;

                    sse_sum += sse;
                    s1v_arr[ch] = s1v;
                    o1v_arr[ch] = o1v;
                }
            }

            double mse = sse_sum / ((double)C * dn);
            if (mse < best_mse) {
                best_mse = mse;
                best_k = k;
                if (quantized) {
                    for (npy_intp ch = 0; ch < C; ch++) {
                        best_s_q[ch] = qs_arr[ch];
                        best_o_q[ch] = qo_arr[ch];
                    }
                } else {
                    for (npy_intp ch = 0; ch < C; ch++) {
                        best_s_f[ch] = s1v_arr[ch];
                        best_o_f[ch] = o1v_arr[ch];
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
    Py_XDECREF(tf_sum);
    Py_XDECREF(tf_sum2);
    Py_XDECREF(cand);
    Py_XDECREF(map_tf);
    Py_XDECREF(map_dom);
    Py_XDECREF(tf_flat);
    Py_XDECREF(img);
    return NULL;
}

PyObject* fastfractal_encode_leaf_best(PyObject* self, PyObject* args) {
    return encode_leaf_best_impl(self,args);
}