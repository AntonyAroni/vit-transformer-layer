#!/bin/bash

echo "Compilando Vision Transformer Training Demo..."

g++ -std=c++17 -I. \
    tests/05_vit_training_demo.cpp \
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
    src/utils/file_io.cpp \
    -o vit_training_demo

if [ $? -eq 0 ]; then
    echo "✅ Compilación exitosa!"
    echo "Ejecutando demo de entrenamiento..."
    ./vit_training_demo
else
    echo "❌ Error en compilación"
fi