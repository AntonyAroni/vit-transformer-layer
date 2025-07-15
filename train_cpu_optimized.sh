#!/bin/bash

echo "Compilando CPU Optimized Vision Transformer..."

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
    -o cpu_optimized_vit

if [ $? -eq 0 ]; then
    echo "✅ Compilación exitosa!"
    echo "Ejecutando entrenamiento optimizado CPU..."
    ./cpu_optimized_vit
else
    echo "❌ Error en compilación"
fi