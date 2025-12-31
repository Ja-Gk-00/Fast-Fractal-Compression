#pragma once

#define PY_SSIZE_T_CLEAN
#include <Python.h>


#define PY_ARRAY_UNIQUE_SYMBOL FASTFRACTAL_ARRAY_API



#ifdef FF_NO_IMPORT_ARRAY
  #define NO_IMPORT_ARRAY
#endif

#include <numpy/arrayobject.h>
#include <stdint.h>

static inline int ff_require_ndim(PyArrayObject *a, int ndim, const char *name) {
    if (PyArray_NDIM(a) != ndim) {
        PyErr_Format(PyExc_ValueError, "%s must be %dD", name, ndim);
        return 0;
    }
    return 1;
}

static inline int ff_require_type(PyArrayObject *a, int typenum, const char *name) {
    if (PyArray_TYPE(a) != typenum) {
        PyErr_Format(PyExc_ValueError, "%s has wrong dtype", name);
        return 0;
    }
    return 1;
}

static inline int ff_require_c_contig(PyArrayObject *a, const char *name) {
    if (!PyArray_IS_C_CONTIGUOUS(a)) {
        PyErr_Format(PyExc_ValueError, "%s must be C-contiguous", name);
        return 0;
    }
    return 1;
}
