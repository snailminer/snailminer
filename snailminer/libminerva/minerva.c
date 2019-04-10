#include <Python.h>
#include <numpy/arrayobject.h>
#include "hash.h"

static PyObject* fruithash(PyObject* self, PyObject* args)
{
    PyArrayObject *array;
    uint64_t * dataset;

    uint8_t mining_hash[DGST_SIZE];
    uint8_t digs[DGST_SIZE];

    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &array)) {
        return Py_None;
    }
    dataset = PyArray_DATA(array);
    fchainhash(dataset, mining_hash, 0, digs);

/*
    {
        int i;
        for (i=0; i < DGST_SIZE; i++) {
            printf("%x", digs[i]);
        }
        printf("\n");
    }
*/
    return Py_None;
}

static PyMethodDef myMethods[] = {
    { "fruithash", fruithash, METH_VARARGS, "calc fruitchain hash" },
    { NULL, NULL, 0, NULL }
};

static struct PyModuleDef myModule = {
    PyModuleDef_HEAD_INIT,
    "truehash",
    "Fruit Module",
    -1,
    myMethods
};

// Initializes our module using our above struct
PyMODINIT_FUNC PyInit_truehash(void)
{
    import_array();
    return PyModule_Create(&myModule);
}
