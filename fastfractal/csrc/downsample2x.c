#define FF_NO_IMPORT_ARRAY
#include "ff_common.h"


PyObject* ff_downsample2x(PyObject* self, PyObject* args) {
    PyObject* img_obj;
    if (!PyArg_ParseTuple(args, "O", &img_obj)) return NULL;

    PyArrayObject* img = (PyArrayObject*)PyArray_FROM_OTF(img_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    if (!img) return NULL;

    int nd = PyArray_NDIM(img);
    if (!(nd == 2 || nd == 3)) {
        PyErr_SetString(PyExc_ValueError, "downsample2x expects HxW or HxWxC float32");
        Py_DECREF(img); return NULL;
    }

    npy_intp H = PyArray_DIM(img, 0);
    npy_intp W = PyArray_DIM(img, 1);
    npy_intp C = (nd == 3) ? PyArray_DIM(img, 2) : 1;

    npy_intp H2 = H / 2;
    npy_intp W2 = W / 2;

    npy_intp out_dims[3];
    int out_nd = nd;
    out_dims[0] = H2;
    out_dims[1] = W2;
    if (nd == 3) out_dims[2] = C;

    PyArrayObject* out = (PyArrayObject*)PyArray_SimpleNew(out_nd, out_dims, NPY_FLOAT32);
    if (!out) { Py_DECREF(img); return NULL; }

    const float* in = (const float*)PyArray_DATA(img);
    float* o = (float*)PyArray_DATA(out);

    
    if (nd == 2) {
        for (npy_intp y = 0; y < H2; y++) {
            for (npy_intp x = 0; x < W2; x++) {
                npy_intp y0 = 2*y, x0 = 2*x;
                float a = in[y0*W + x0];
                float b = in[y0*W + (x0+1)];
                float c = in[(y0+1)*W + x0];
                float d = in[(y0+1)*W + (x0+1)];
                o[y*W2 + x] = 0.25f * (a + b + c + d);
            }
        }
    } else {
        npy_intp strideW = C;
        npy_intp strideH = W * C;
        npy_intp strideOW = C;
        npy_intp strideOH = W2 * C;

        for (npy_intp y = 0; y < H2; y++) {
            for (npy_intp x = 0; x < W2; x++) {
                npy_intp y0 = 2*y, x0 = 2*x;
                const float* p00 = in + y0*strideH + x0*strideW;
                const float* p01 = p00 + strideW;
                const float* p10 = p00 + strideH;
                const float* p11 = p10 + strideW;
                float* po = o + y*strideOH + x*strideOW;
                for (npy_intp cidx = 0; cidx < C; cidx++) {
                    po[cidx] = 0.25f * (p00[cidx] + p01[cidx] + p10[cidx] + p11[cidx]);
                }
            }
        }
    }

    Py_DECREF(img);
    return (PyObject*)out;
}
