#pragma once
#include "../matrix/matrix.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>

class CudaMatrix {
private:
    float* d_data;
    int rows, cols;
    static cublasHandle_t cublas_handle;

public:
    CudaMatrix(int rows, int cols);
    CudaMatrix(const Matrix& host_matrix);
    ~CudaMatrix();
    
    void copyToHost(Matrix& host_matrix);
    void copyFromHost(const Matrix& host_matrix);
    
    static CudaMatrix matmul(const CudaMatrix& a, const CudaMatrix& b);
    static void initCublas();
    static void destroyCublas();
    
    float* getData() const { return d_data; }
    int getRows() const { return rows; }
    int getCols() const { return cols; }
};
#endif