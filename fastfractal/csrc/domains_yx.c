#define FF_NO_IMPORT_ARRAY
#include "ff_common.h"


PyObject* ff_domains_yx(PyObject* self, PyObject* args) {
    int h, w, block, stride;
    if (!PyArg_ParseTuple(args, "iiii", &h, &w, &block, &stride)) return NULL;

    if (h <= 0 || w <= 0 || block <= 0 || stride <= 0) {
        PyErr_SetString(PyExc_ValueError, "h,w,block,stride must be > 0");
        return NULL;
    }

    const int dom_h = h - 2 * block + 1;
    const int dom_w = w - 2 * block + 1;
    if (dom_h <= 0 || dom_w <= 0) {
        npy_intp dims[2] = {0, 2};
        return (PyObject*)PyArray_SimpleNew(2, dims, NPY_UINT16);
    }

    const npy_intp ny = (dom_h + stride - 1) / stride;
    const npy_intp nx = (dom_w + stride - 1) / stride;
    npy_intp out_dims[2] = {ny * nx, 2};

    PyArrayObject* out = (PyArrayObject*)PyArray_SimpleNew(2, out_dims, NPY_UINT16);
    if (!out) return NULL;

    uint16_t* p = (uint16_t*)PyArray_DATA(out);

    npy_intp idx = 0;
    for (int y = 0; y < dom_h; y += stride) {
        for (int x = 0; x < dom_w; x += stride) {
            
            p[2 * idx + 0] = (uint16_t)y;
            p[2 * idx + 1] = (uint16_t)x;
            idx++;
        }
    }

    return (PyObject*)out;
}
