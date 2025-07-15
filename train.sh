#!/bin/bash

echo "=== COMPILANDO ENTRENAMIENTO TRANSFORMER ==="
g++ -std=c++17 -I. tests/04_train_fashion_mnist.cpp \
    src/matrix/matrix.cpp \
    src/matrix/matrix_ops.cpp \
    src/matrix/activation_functions.h.cpp \
    src/transformer/multi_head_attention.cpp \
    src/transformer/mlp.cpp \
    src/transformer/layer_norm.cpp \
    src/transformer/transformer_block.cpp \
    src/utils/file_io.cpp \
    src/training/trainer.cpp \
    -o train_fashion

if [ $? -eq 0 ]; then
    echo "✅ Compilación exitosa!"
    echo ""
    echo "=== EJECUTANDO ENTRENAMIENTO ==="
    ./train_fashion
else
    echo "❌ Error en compilación"
    exit 1
fi