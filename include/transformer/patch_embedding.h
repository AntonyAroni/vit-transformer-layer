#pragma once
#include "../matrix/matrix.h"

class PatchEmbedding {
private:
    int patch_size;
    int embed_dim;
    Matrix projection_weight;
    Matrix projection_bias;

public:
    PatchEmbedding(int patch_size, int embed_dim);
    Matrix forward(const Matrix& images);
    int get_num_patches(int img_size) const;
};