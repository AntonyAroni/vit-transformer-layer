#!/bin/bash

echo "Compilando Vision Transformer Training..."

g++ -std=c++17 -I. \
    tests/06_vit_training.cpp \
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
    src/training/optimizer.cpp \
    src/utils/file_io.cpp \
    -o vit_training

if [ $? -eq 0 ]; then
    echo "✅ Compilación exitosa!"
    echo "Ejecutando entrenamiento..."
    ./vit_training
else
    echo "❌ Error en compilación"
fi