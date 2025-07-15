#ifdef USE_CUDA
#include "../../include/cuda/cuda_matrix.h"
#include <iostream>

cublasHandle_t CudaMatrix::cublas_handle;

CudaMatrix::CudaMatrix(int rows, int cols) : rows(rows), cols(cols) {
    cudaMalloc(&d_data, rows * cols * sizeof(float));
}

CudaMatrix::CudaMatrix(const Matrix& host_matrix) : rows(host_matrix.getRows()), cols(host_matrix.getCols()) {
    cudaMalloc(&d_data, rows * cols * sizeof(float));
    copyFromHost(host_matrix);
}

CudaMatrix::~CudaMatrix() {
    cudaFree(d_data);
}

void CudaMatrix::copyFromHost(const Matrix& host_matrix) {
    float* temp = new float[rows * cols];
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            temp[i * cols + j] = (float)host_matrix(i, j);
        }
    }
    cudaMemcpy(d_data, temp, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    delete[] temp;
}

void CudaMatrix::copyToHost(Matrix& host_matrix) {
    float* temp = new float[rows * cols];
    cudaMemcpy(temp, d_data, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            host_matrix(i, j) = (double)temp[i * cols + j];
        }
    }
    delete[] temp;
}

CudaMatrix CudaMatrix::matmul(const CudaMatrix& a, const CudaMatrix& b) {
    CudaMatrix result(a.rows, b.cols);
    const float alpha = 1.0f, beta = 0.0f;
    
    cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                b.cols, a.rows, a.cols,
                &alpha, b.d_data, b.cols,
                a.d_data, a.cols,
                &beta, result.d_data, b.cols);
    
    return result;
}

void CudaMatrix::initCublas() {
    cublasCreate(&cublas_handle);
}

void CudaMatrix::destroyCublas() {
    cublasDestroy(cublas_handle);
}
#endif