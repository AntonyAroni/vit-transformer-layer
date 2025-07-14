#include "../../include/transformer/multi_head_attention.h"
#include "../../include/matrix/matrix_ops.h"
#include "../../include/matrix/activation_functions.h"
#include <cmath>

MultiHeadAttention::MultiHeadAttention(size_t embed_dim, size_t num_heads) 
    : embed_dim(embed_dim), num_heads(num_heads) {
    
    if (embed_dim % num_heads != 0) {
        throw std::runtime_error("embed_dim must be divisible by num_heads");
    }
    
    head_dim = embed_dim / num_heads;
    initialize_weights();
}

void MultiHeadAttention::initialize_weights() {
    // Xavier initialization
    double scale = sqrt(2.0 / embed_dim);
    
    W_q = Matrix::random(embed_dim, embed_dim) * scale;
    W_k = Matrix::random(embed_dim, embed_dim) * scale;
    W_v = Matrix::random(embed_dim, embed_dim) * scale;
    W_o = Matrix::random(embed_dim, embed_dim) * scale;
}

Matrix MultiHeadAttention::scaled_dot_product_attention(const Matrix& Q, const Matrix& K, const Matrix& V) {
    // Q, K, V: [seq_len, head_dim]
    Matrix K_T = MatrixOps::transpose(K);
    Matrix scores = MatrixOps::matmul(Q, K_T);
    
    // Scale by sqrt(head_dim)
    double scale = 1.0 / sqrt(head_dim);
    scores = scores * scale;
    
    // Apply softmax to each row
    Matrix attention_weights = ActivationFunctions::softmax(scores);
    
    // Apply attention to values
    return MatrixOps::matmul(attention_weights, V);
}

Matrix MultiHeadAttention::forward(const Matrix& input) {
    size_t seq_len = input.getRows();
    
    // Linear projections
    Matrix Q = MatrixOps::matmul(input, W_q);
    Matrix K = MatrixOps::matmul(input, W_k);
    Matrix V = MatrixOps::matmul(input, W_v);

    
    // Split into multiple heads and compute attention
    Matrix output = Matrix::zeros(seq_len, embed_dim);
    
    for (size_t h = 0; h < num_heads; ++h) {
        size_t start_col = h * head_dim;
        
        // Extract head-specific Q, K, V
        Matrix Q_h = Matrix::zeros(seq_len, head_dim);
        Matrix K_h = Matrix::zeros(seq_len, head_dim);
        Matrix V_h = Matrix::zeros(seq_len, head_dim);
        
        for (size_t i = 0; i < seq_len; ++i) {
            for (size_t j = 0; j < head_dim; ++j) {
                Q_h(i, j) = Q(i, start_col + j);
                K_h(i, j) = K(i, start_col + j);
                V_h(i, j) = V(i, start_col + j);
            }
        }
        
        // Compute attention for this head
        Matrix head_output = scaled_dot_product_attention(Q_h, K_h, V_h);
        
        // Place back into output
        for (size_t i = 0; i < seq_len; ++i) {
            for (size_t j = 0; j < head_dim; ++j) {
                output(i, start_col + j) = head_output(i, j);
            }
        }
    }
    
    // Final linear projection
    return MatrixOps::matmul(output, W_o);
}