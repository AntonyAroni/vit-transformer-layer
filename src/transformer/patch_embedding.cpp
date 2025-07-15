#include "../../include/transformer/patch_embedding.h"
#include "../../include/matrix/matrix_ops.h"
#include <cmath>

PatchEmbedding::PatchEmbedding(int patch_size, int embed_dim) 
    : patch_size(patch_size), embed_dim(embed_dim) {
    int patch_dim = patch_size * patch_size;
    projection_weight = Matrix(embed_dim, patch_dim);
    projection_bias = Matrix(embed_dim, 1);
    
    // Xavier initialization
    double std = sqrt(2.0 / (patch_dim + embed_dim));
    for (int i = 0; i < embed_dim; i++) {
        for (int j = 0; j < patch_dim; j++) {
            projection_weight(i, j) = ((double)rand() / RAND_MAX - 0.5) * 2 * std;
        }
        projection_bias(i, 0) = 0.0;
    }
}

Matrix PatchEmbedding::forward(const Matrix& images) {
    int batch_size = images.getRows();
    int img_size = (int)sqrt(images.getCols());
    int num_patches = get_num_patches(img_size);
    
    Matrix patches(batch_size * num_patches, patch_size * patch_size);
    
    // Extract patches
    int patch_idx = 0;
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < img_size; i += patch_size) {
            for (int j = 0; j < img_size; j += patch_size) {
                for (int pi = 0; pi < patch_size; pi++) {
                    for (int pj = 0; pj < patch_size; pj++) {
                        int img_row = i + pi;
                        int img_col = j + pj;
                        if (img_row < img_size && img_col < img_size) {
                            patches(patch_idx, pi * patch_size + pj) = 
                                images(b, img_row * img_size + img_col);
                        }
                    }
                }
                patch_idx++;
            }
        }
    }
    
    // Project patches to embedding dimension
    Matrix embeddings = MatrixOps::matmul(patches, MatrixOps::transpose(projection_weight));
    
    // Add bias
    for (int i = 0; i < embeddings.getRows(); i++) {
        for (int j = 0; j < embeddings.getCols(); j++) {
            embeddings(i, j) += projection_bias(j, 0);
        }
    }
    
    return embeddings;
}

int PatchEmbedding::get_num_patches(int img_size) const {
    return (img_size / patch_size) * (img_size / patch_size);
}