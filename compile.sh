#!/bin/bash

echo "Compilando Complete Vision Transformer..."

g++ -std=c++17 -I. \
    tests/04_complete_vit_test.cpp \
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
    src/utils/file_io.cpp \
    -o complete_vit_test

if [ $? -eq 0 ]; then
    echo "✅ Compilación exitosa!"
    echo "Ejecutando test..."
    ./complete_vit_test
else
    echo "❌ Error en compilación"
fi