  #!/bin/bash

echo "Compilando Optimized Vision Transformer..."

# Add CUDA to PATH
export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH

# Check for CUDA
if command -v nvcc &> /dev/null; then
    echo "CUDA detected, compiling with GPU support..."
    nvcc -std=c++17 -arch=sm_75 -I. -DUSE_CUDA \
        tests/07_optimized_training.cpp \
        src/matrix/matrix.cpp \
        src/matrix/matrix_ops.cpp \
        src/matrix/activation_functions.h.cpp \
        src/transformer/multi_head_attention.cpp \
        src/transformer/mlp.cpp \
        src/transformer/layer_norm.cpp \
        src/transformer/transformer_block.cpp \
        src/transformer/patch_embedding.cpp \
        src/transformer/positional_encoding.cpp \
        src/transformer/vision_transformer.cpp \
        src/transformer/loss_functions.cpp \
        src/training/adam_optimizer.cpp \
        src/training/lr_scheduler.cpp \
        src/utils/data_augmentation.cpp \
        src/utils/file_io.cpp \
        src/cuda/cuda_matrix.cu \
        -lcublas -lcudart \
        -o optimized_vit_training
else
    echo "CUDA not found, compiling CPU version..."
    g++ -std=c++17 -I. -O3 -march=native -fopenmp \
        tests/07_optimized_training.cpp \
        src/matrix/matrix.cpp \
        src/matrix/matrix_ops.cpp \
        src/matrix/activation_functions.h.cpp \
        src/transformer/multi_head_attention.cpp \
        src/transformer/mlp.cpp \
        src/transformer/layer_norm.cpp \
        src/transformer/transformer_block.cpp \
        src/transformer/patch_embedding.cpp \
        src/transformer/positional_encoding.cpp \
        src/transformer/vision_transformer.cpp \
        src/transformer/loss_functions.cpp \
        src/training/adam_optimizer.cpp \
        src/training/lr_scheduler.cpp \
        src/utils/data_augmentation.cpp \
        src/utils/file_io.cpp \
        -fopenmp \
        -o optimized_vit_training
fi

if [ $? -eq 0 ]; then
    echo "✅ Compilación exitosa!"
    echo "Ejecutando entrenamiento optimizado..."
    ./optimized_vit_training
else
    echo "❌ Error en compilación"
fi