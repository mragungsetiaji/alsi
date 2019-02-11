
#ifndef ALSI_BPR_H_
#define ALSI_BPR_H_

#ifdef _OPENMP
#include <omp.h>
#endif

namespace alsi {
#ifdef _OPENMP
inline int get_thread_num() { return omp_get_thread_num(); }
#else
inline int get_thread_num() { return 0; }
#endif
} 
#endif