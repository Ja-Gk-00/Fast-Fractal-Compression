#define FF_NO_IMPORT_ARRAY
#include "ff_common.h"


static inline void map_tf(int tf, int b, int dy, int dx, int* sy, int* sx) {
    switch (tf & 7) {
        case 0: *sy = dy;         *sx = dx;         break; 
        case 1: *sy = dy;         *sx = b - 1 - dx; break; 
        case 2: *sy = b - 1 - dy; *sx = dx;         break; 
        case 3: *sy = b - 1 - dy; *sx = b - 1 - dx; break; 
        case 4: *sy = dx;         *sx = dy;         break; 
        case 5: *sy = dx;         *sx = b - 1 - dy; break;
        case 6: *sy = b - 1 - dx; *sx = dy;         break;
        case 7: *sy = b - 1 - dx; *sx = b - 1 - dy; break;
    }
}

PyObject* ff_decode_apply(PyObject* self, PyObject* args) {
    PyObject *dst_obj, *src_obj, *leaf_yx_obj, *dom_yx_obj, *leaf_block_obj, *leaf_tf_obj, *s_obj, *o_obj;
    if (!PyArg_ParseTuple(
            args, "OOOOOOOO",
            &dst_obj, &src_obj, &leaf_yx_obj, &dom_yx_obj, &leaf_block_obj, &leaf_tf_obj, &s_obj, &o_obj
        )) return NULL;

    PyArrayObject* dst = (PyArrayObject*)PyArray_FROM_OTF(dst_obj, NPY_FLOAT32, NPY_ARRAY_INOUT_ARRAY2);
    PyArrayObject* src = (PyArrayObject*)PyArray_FROM_OTF(src_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* leaf_yx = (PyArrayObject*)PyArray_FROM_OTF(leaf_yx_obj, NPY_UINT16, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* dom_yx  = (PyArrayObject*)PyArray_FROM_OTF(dom_yx_obj,  NPY_UINT16, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* leaf_block = (PyArrayObject*)PyArray_FROM_OTF(leaf_block_obj, NPY_UINT16, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* leaf_tf = (PyArrayObject*)PyArray_FROM_OTF(leaf_tf_obj, NPY_UINT8, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* s = (PyArrayObject*)PyArray_FROM_OTF(s_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* o = (PyArrayObject*)PyArray_FROM_OTF(o_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);

    if (!dst || !src || !leaf_yx || !dom_yx || !leaf_block || !leaf_tf || !s || !o) {
        Py_XDECREF(src); Py_XDECREF(leaf_yx); Py_XDECREF(dom_yx); Py_XDECREF(leaf_block);
        Py_XDECREF(leaf_tf); Py_XDECREF(s); Py_XDECREF(o);
        if (dst) PyArray_DiscardWritebackIfCopy(dst), Py_DECREF(dst);
        return NULL;
    }

    int nd = PyArray_NDIM(dst);
    if (!(nd == 2 || nd == 3)) {
        PyErr_SetString(PyExc_ValueError, "dst must be HxW or HxWxC float32");
        goto fail;
    }
    if (PyArray_NDIM(src) != nd) {
        PyErr_SetString(PyExc_ValueError, "src and dst must have same dimensionality");
        goto fail;
    }

    npy_intp H = PyArray_DIM(dst, 0);
    npy_intp W = PyArray_DIM(dst, 1);
    npy_intp C = (nd == 3) ? PyArray_DIM(dst, 2) : 1;

    if (nd == 3 && PyArray_DIM(src, 2) != C) {
        PyErr_SetString(PyExc_ValueError, "src and dst must have same channel count");
        goto fail;
    }

    if (PyArray_NDIM(leaf_yx) != 2 || PyArray_DIM(leaf_yx, 1) != 2) {
        PyErr_SetString(PyExc_ValueError, "leaf_yx must be (L,2) uint16");
        goto fail;
    }
    npy_intp L = PyArray_DIM(leaf_yx, 0);

    if (PyArray_NDIM(dom_yx) != 2 || PyArray_DIM(dom_yx, 0) != L || PyArray_DIM(dom_yx, 1) != 2) {
        PyErr_SetString(PyExc_ValueError, "dom_yx must be (L,2) uint16");
        goto fail;
    }
    if (PyArray_DIM(leaf_block, 0) != L || PyArray_DIM(leaf_tf, 0) != L || PyArray_DIM(s, 0) != L || PyArray_DIM(o, 0) != L) {
        PyErr_SetString(PyExc_ValueError, "leaf_block/leaf_tf/s/o must all have length L");
        goto fail;
    }

    float* Dst = (float*)PyArray_DATA(dst);
    const float* Src = (const float*)PyArray_DATA(src);

    const unsigned short* LY = (const unsigned short*)PyArray_DATA(leaf_yx);
    const unsigned short* DY = (const unsigned short*)PyArray_DATA(dom_yx);
    const unsigned short* LB = (const unsigned short*)PyArray_DATA(leaf_block);
    const unsigned char* TF = (const unsigned char*)PyArray_DATA(leaf_tf);
    const float* S = (const float*)PyArray_DATA(s);
    const float* O = (const float*)PyArray_DATA(o);

    npy_intp strideW = C;
    npy_intp strideH = W * C;

    for (npy_intp i = 0; i < L; i++) {
        int y0 = (int)LY[2*i + 0];
        int x0 = (int)LY[2*i + 1];
        int yd = (int)DY[2*i + 0];
        int xd = (int)DY[2*i + 1];
        int b  = (int)LB[i];
        int tf = (int)TF[i];
        float ss = S[i];
        float oo = O[i];

        
        if (y0 < 0 || x0 < 0 || yd < 0 || xd < 0) continue;
        if (y0 + b > (int)H || x0 + b > (int)W) continue;
        if (yd + b > (int)H || xd + b > (int)W) continue;

        for (int dy = 0; dy < b; dy++) {
            for (int dx = 0; dx < b; dx++) {
                int sy, sx;
                map_tf(tf, b, dy, dx, &sy, &sx);
                npy_intp src_off = (npy_intp)(yd + sy) * strideH + (npy_intp)(xd + sx) * strideW;
                npy_intp dst_off = (npy_intp)(y0 + dy) * strideH + (npy_intp)(x0 + dx) * strideW;

                if (C == 1) {
                    Dst[dst_off] = ss * Src[src_off] + oo;
                } else {
                    for (npy_intp c = 0; c < C; c++) {
                        Dst[dst_off + c] = ss * Src[src_off + c] + oo;
                    }
                }
            }
        }
    }

    PyArray_ResolveWritebackIfCopy(dst);
    Py_DECREF(dst);
    Py_DECREF(src);
    Py_DECREF(leaf_yx);
    Py_DECREF(dom_yx);
    Py_DECREF(leaf_block);
    Py_DECREF(leaf_tf);
    Py_DECREF(s);
    Py_DECREF(o);
    Py_RETURN_NONE;

fail:
    PyArray_DiscardWritebackIfCopy(dst);
    Py_DECREF(dst);
    Py_DECREF(src);
    Py_DECREF(leaf_yx);
    Py_DECREF(dom_yx);
    Py_DECREF(leaf_block);
    Py_DECREF(leaf_tf);
    Py_DECREF(s);
    Py_DECREF(o);
    return NULL;
}
