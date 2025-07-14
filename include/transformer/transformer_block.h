#ifndef TRANSFORMER_BLOCK_H
#define TRANSFORMER_BLOCK_H

#include "../matrix/matrix.h"
#include "multi_head_attention.h"
#include "mlp.h"
#include "layer_norm.h"

class TransformerBlock {
private:
    MultiHeadAttention attention;
    MLP mlp;
    LayerNorm norm1, norm2;
    
public:
    TransformerBlock(size_t embed_dim, size_t num_heads, size_t mlp_hidden_dim);
    
    Matrix forward(const Matrix& input);
};

#endif