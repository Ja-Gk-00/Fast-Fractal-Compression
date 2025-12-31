#define FF_NO_IMPORT_ARRAY
#include "ff_common.h"


PyObject* ff_extract_range(PyObject* self, PyObject* args) {
    PyObject* img_obj = NULL;
    int y, x, b;
    if (!PyArg_ParseTuple(args, "Oiii", &img_obj, &y, &x, &b)) return NULL;

    PyArrayObject* img = (PyArrayObject*)PyArray_FROM_OTF(img_obj, NPY_FLOAT32, NPY_ARRAY_ALIGNED);
    if (!img) return NULL;

    int ndim = PyArray_NDIM(img);
    if (ndim != 2 && ndim != 3) {
        PyErr_SetString(PyExc_ValueError, "img must be 2D or 3D float32");
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
    npy_intp C = (ndim == 3) ? PyArray_DIM(img, 2) : 1;

    if (y < 0 || x < 0 || (npy_intp)(y + b) > H || (npy_intp)(x + b) > W) {
        PyErr_SetString(PyExc_ValueError, "extract_range: (y,x,block) out of bounds");
        Py_DECREF(img);
        return NULL;
    }

    
    npy_intp odims[3];
    npy_intp ostrides[3];

    odims[0] = (npy_intp)b;
    odims[1] = (npy_intp)b;

    ostrides[0] = PyArray_STRIDE(img, 0);
    ostrides[1] = PyArray_STRIDE(img, 1);

    if (ndim == 3) {
        odims[2] = C;
        ostrides[2] = PyArray_STRIDE(img, 2);
    }

    char* base = (char*)PyArray_DATA(img);
    char* data_ptr = base + (npy_intp)y * PyArray_STRIDE(img, 0) + (npy_intp)x * PyArray_STRIDE(img, 1);

    PyArray_Descr* descr = PyArray_DESCR(img);
    Py_INCREF(descr);

    PyObject* out = PyArray_NewFromDescr(
        &PyArray_Type,
        descr,
        ndim,
        odims,
        ostrides,
        (void*)data_ptr,
        0,
        (PyObject*)img
    );

    
    Py_DECREF(img);
    return out;
}
