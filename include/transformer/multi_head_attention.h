#ifndef MULTI_HEAD_ATTENTION_H
#define MULTI_HEAD_ATTENTION_H

#include "../matrix/matrix.h"

class MultiHeadAttention {
private:
    size_t embed_dim;
    size_t num_heads;
    size_t head_dim;
    
    Matrix W_q, W_k, W_v, W_o;  // Weight matrices
    
public:
    MultiHeadAttention(size_t embed_dim, size_t num_heads);
    
    Matrix forward(const Matrix& input);
    Matrix scaled_dot_product_attention(const Matrix& Q, const Matrix& K, const Matrix& V);
    
    void initialize_weights();
};

#endif