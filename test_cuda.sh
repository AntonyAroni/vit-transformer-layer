#!/bin/bash

echo "Compilando CUDA Test..."

# Add CUDA to PATH
export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH

nvcc -std=c++17 -arch=sm_75 \
    tests/cuda_test.cu \
    -lcublas -lcudart \
    -o cuda_test

if [ $? -eq 0 ]; then
    echo "✅ Compilación CUDA exitosa!"
    echo "Ejecutando test CUDA..."
    ./cuda_test
else
    echo "❌ Error en compilación CUDA"
    echo "Verificando instalación..."
    echo "NVCC version:"
    nvcc --version
    echo "NVIDIA driver:"
    nvidia-smi
fi