import logging
import os
import numpy as np

def nonzeros(m, row):
    for index in range(m.indptr[row], m.indptr[row+1]):
        yield m.indices[index], m.data[index]

_checked_blas_config = False
def check_blas_config():

    global _checked_blas_config
    if _checked_blas_config:
        return
    _checked_blas_config = True

    if np.__config__.get_info('openblas_info') and os.environ.get('OPENBLAS_NUM_THREADS') != '1':
        logging.warning("OpenBLAS detected. Its highly recommend to set the environment variable "
                        "'export OPENBLAS_NUM_THREADS=1' to disable its internal multithreading")
    if np.__config__.get_info('blas_mkl_info') and os.environ.get('MKL_NUM_THREADS') != '1':
        logging.warning("Intel MKL BLAS detected. Its highly recommend to set the environment "
                        "variable 'export MKL_NUM_THREADS=1' to disable its internal "
                        "multithreading")
