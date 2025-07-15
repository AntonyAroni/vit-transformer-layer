#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>

__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    std::cout << "=== CUDA TEST ===" << std::endl;
    
    // Check CUDA devices
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    std::cout << "CUDA devices found: " << deviceCount << std::endl;
    
    if (deviceCount == 0) {
        std::cout << "❌ No CUDA devices found!" << std::endl;
        return -1;
    }
    
    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Device 0: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Global memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
    
    // Test basic CUDA kernel
    const int N = 1024;
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    
    // Allocate host memory
    h_a = new float[N];
    h_b = new float[N];
    h_c = new float[N];
    
    // Initialize arrays
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }
    
    // Allocate device memory
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));
    
    // Copy to device
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    std::cout << "\nLaunching CUDA kernel..." << std::endl;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "❌ CUDA kernel error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    
    // Copy result back
    cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Verify results
    bool success = true;
    for (int i = 0; i < 10; i++) {
        float expected = h_a[i] + h_b[i];
        if (abs(h_c[i] - expected) > 1e-5) {
            success = false;
            break;
        }
    }
    
    if (success) {
        std::cout << "✅ CUDA kernel test passed!" << std::endl;
    } else {
        std::cout << "❌ CUDA kernel test failed!" << std::endl;
    }
    
    // Test cuBLAS
    std::cout << "\nTesting cuBLAS..." << std::endl;
    cublasHandle_t handle;
    cublasStatus_t stat = cublasCreate(&handle);
    
    if (stat == CUBLAS_STATUS_SUCCESS) {
        std::cout << "✅ cuBLAS initialized successfully!" << std::endl;
        
        // Simple matrix multiplication test
        const int M = 64, N = 64, K = 64;
        float *d_A, *d_B, *d_C;
        
        cudaMalloc(&d_A, M * K * sizeof(float));
        cudaMalloc(&d_B, K * N * sizeof(float));
        cudaMalloc(&d_C, M * N * sizeof(float));
        
        // Initialize with ones
        float *h_ones = new float[M * K];
        for (int i = 0; i < M * K; i++) h_ones[i] = 1.0f;
        
        cudaMemcpy(d_A, h_ones, M * K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_ones, K * N * sizeof(float), cudaMemcpyHostToDevice);
        
        const float alpha = 1.0f, beta = 0.0f;
        stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                          N, M, K, &alpha,
                          d_B, N, d_A, K,
                          &beta, d_C, N);
        
        if (stat == CUBLAS_STATUS_SUCCESS) {
            std::cout << "✅ cuBLAS matrix multiplication successful!" << std::endl;
        } else {
            std::cout << "❌ cuBLAS matrix multiplication failed!" << std::endl;
        }
        
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        delete[] h_ones;
        
        cublasDestroy(handle);
    } else {
        std::cout << "❌ cuBLAS initialization failed!" << std::endl;
    }
    
    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    
    std::cout << "\n✅ CUDA test completed!" << std::endl;
    
    return 0;
}