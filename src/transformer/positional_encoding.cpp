#include "../../include/transformer/positional_encoding.h"
#include <cmath>

PositionalEncoding::PositionalEncoding(int max_seq_len, int embed_dim) 
    : max_seq_len(max_seq_len), embed_dim(embed_dim) {
    pos_embedding = Matrix(max_seq_len, embed_dim);
    
    // Learnable positional embeddings (random initialization)
    double std = 0.02;
    for (int i = 0; i < max_seq_len; i++) {
        for (int j = 0; j < embed_dim; j++) {
            pos_embedding(i, j) = ((double)rand() / RAND_MAX - 0.5) * 2 * std;
        }
    }
}

Matrix PositionalEncoding::forward(const Matrix& x) {
    Matrix result = x;
    int seq_len = x.getRows();
    
    // Add positional embeddings
    for (int i = 0; i < seq_len && i < max_seq_len; i++) {
        for (int j = 0; j < x.getCols(); j++) {
            result(i, j) += pos_embedding(i, j);
        }
    }
    
    return result;
}