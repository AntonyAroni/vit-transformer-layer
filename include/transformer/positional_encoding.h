#pragma once
#include "../matrix/matrix.h"

class PositionalEncoding {
private:
    Matrix pos_embedding;
    int max_seq_len;
    int embed_dim;

public:
    PositionalEncoding(int max_seq_len, int embed_dim);
    Matrix forward(const Matrix& x);
};