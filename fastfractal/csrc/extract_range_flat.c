#define FF_NO_IMPORT_ARRAY
#include "ff_common.h"

#include <string.h>

PyObject* ff_extract_range_flat(PyObject* self, PyObject* args) {
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
    if (!out) {
        Py_DECREF(img);
        return NULL;
    }

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
