#pragma once
#include "../matrix/matrix.h"
#include "patch_embedding.h"
#include "positional_encoding.h"
#include "transformer_block.h"
#include <vector>

class VisionTransformer {
private:
    PatchEmbedding patch_embed;
    PositionalEncoding pos_encoding;
    std::vector<TransformerBlock> transformer_blocks;
    Matrix cls_token;
    Matrix classification_head_weight;
    Matrix classification_head_bias;
    
    int embed_dim;
    int num_classes;
    int num_layers;
    double dropout_rate;
    bool use_cuda;

public:
    VisionTransformer(int img_size, int patch_size, int embed_dim, 
                     int num_heads, int mlp_dim, int num_layers, int num_classes, 
                     double dropout = 0.1, bool cuda = false);
    Matrix forward(const Matrix& images, bool training = true);
    Matrix get_predictions(const Matrix& logits);
    void backward_and_update(const Matrix& images, const std::vector<int>& labels, double learning_rate);
    void setTraining(bool training) { /* for dropout */ }
    void initWeights();
};