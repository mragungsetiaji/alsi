#ifndef ALSI_CUDA_BPR_H_
#define ALSI_CUDA_BPR_H_
#include "alsi/cuda/matrix.h"
#include <utility>

namespace alsi {
std::pair<int, int>  bpr_update(const CudaVector<int> & userids,
                                const CudaVector<int> & itemids,
                                const CudaVector<int> & indptr,
                                CudaDenseMatrix * X,
                                CudaDenseMatrix * Y,
                                float learning_rate,
                                float regularization,
                                long seed,
                                bool verify_negative_samples);
}  
#endif  