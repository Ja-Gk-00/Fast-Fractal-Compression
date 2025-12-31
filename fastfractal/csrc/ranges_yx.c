#define FF_NO_IMPORT_ARRAY
#include "ff_common.h"


PyObject* ff_ranges_yx(PyObject* self, PyObject* args) {
    int h, w, block;
    if (!PyArg_ParseTuple(args, "iii", &h, &w, &block)) return NULL;

    if (h <= 0 || w <= 0 || block <= 0) {
        PyErr_SetString(PyExc_ValueError, "h,w,block must be positive");
        return NULL;
    }

    int lim_y = h - block;
    int lim_x = w - block;

    npy_intp dims[2] = {0, 2};
    if (lim_y < 0 || lim_x < 0) {
        return (PyObject*)PyArray_SimpleNew(2, dims, NPY_UINT16);
    }

    int ny = lim_y / block + 1;
    int nx = lim_x / block + 1;
    npy_intp M = (npy_intp)ny * (npy_intp)nx;

    dims[0] = M;
    PyArrayObject* out = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_UINT16);
    if (!out) return NULL;

    uint16_t* p = (uint16_t*)PyArray_DATA(out);
    npy_intp idx = 0;
    for (int y = 0; y <= lim_y; y += block) {
        for (int x = 0; x <= lim_x; x += block) {
            p[2 * idx + 0] = (uint16_t)y;
            p[2 * idx + 1] = (uint16_t)x;
            idx++;
        }
    }

    return (PyObject*)out;
}
