#ifndef ALSI_CUDA_ALS_H_
#define ALSI_CUDA_ALS_H_
#include "alsi/cuda/matrix.h"

struct cublasContext;

namespace alsi {

struct CudaLeastSquaresSolver {
    explicit CudaLeastSquaresSolver(int factors);
    ~CudaLeastSquaresSolver();

    void least_squares(const CudaCSRMatrix & Cui,
                       CudaDenseMatrix * X, const CudaDenseMatrix & Y,
                       float regularization,
                       int cg_steps) const;

    float calculate_loss(const CudaCSRMatrix & Cui,
                        const CudaDenseMatrix & X,
                        const CudaDenseMatrix & Y,
                        float regularization);

    CudaDenseMatrix YtY;
    cublasContext * blas_handle;
};
}  
#endif 