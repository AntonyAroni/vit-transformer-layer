#!/bin/bash

echo "Compilando Transformer Layer..."

g++ -std=c++17 -I. \
    tests/test_transformer_layer.cpp \
    src/matrix/matrix.cpp \
    src/matrix/matrix_ops.cpp \
    src/matrix/activation_functions.h.cpp \
    src/transformer/multi_head_attention.cpp \
    src/transformer/mlp.cpp \
    src/transformer/layer_norm.cpp \
    src/transformer/transformer_block.cpp \
    -o test_transformer

if [ $? -eq 0 ]; then
    echo "✅ Compilación exitosa!"
    echo "Ejecutando test..."
    ./test_transformer
else
    echo "❌ Error en compilación"
fi