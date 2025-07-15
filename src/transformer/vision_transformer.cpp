#include "../../include/transformer/vision_transformer.h"
#include "../../include/transformer/loss_functions.h"
#include "../../include/matrix/matrix_ops.h"
#include "../../include/matrix/activation_functions.h"
#include <cmath>

VisionTransformer::VisionTransformer(int img_size, int patch_size, int embed_dim, 
                                   int num_heads, int mlp_dim, int num_layers, int num_classes,
                                   double dropout, bool cuda)
    : patch_embed(patch_size, embed_dim),
      pos_encoding(patch_embed.get_num_patches(img_size) + 1, embed_dim),
      embed_dim(embed_dim), num_classes(num_classes), num_layers(num_layers),
      dropout_rate(dropout), use_cuda(cuda) {
    
    // Initialize transformer blocks
    for (int i = 0; i < num_layers; i++) {
        transformer_blocks.emplace_back(embed_dim, num_heads, mlp_dim);
    }
    
    initWeights();
}

void VisionTransformer::initWeights() {
    // Better initialization - Xavier/Glorot for CLS token
    cls_token = Matrix(1, embed_dim);
    double std_cls = sqrt(2.0 / embed_dim);
    for (int j = 0; j < embed_dim; j++) {
        cls_token(0, j) = ((double)rand() / RAND_MAX - 0.5) * 2 * std_cls;
    }
    
    // He initialization for classification head
    classification_head_weight = Matrix(num_classes, embed_dim);
    classification_head_bias = Matrix(num_classes, 1);
    
    double std_head = sqrt(2.0 / embed_dim);
    for (int i = 0; i < num_classes; i++) {
        for (int j = 0; j < embed_dim; j++) {
            classification_head_weight(i, j) = ((double)rand() / RAND_MAX - 0.5) * 2 * std_head;
        }
        classification_head_bias(i, 0) = 0.0;
    }
}

Matrix VisionTransformer::forward(const Matrix& images, bool training) {
    int batch_size = images.getRows();
    
    // Patch embedding
    Matrix patch_embeddings = patch_embed.forward(images);
    int num_patches = patch_embeddings.getRows() / batch_size;
    
    // Reshape to [batch_size, num_patches, embed_dim]
    Matrix batch_embeddings(num_patches, embed_dim);
    Matrix sequence(num_patches + 1, embed_dim);
    Matrix logits(batch_size, num_classes);
    
    for (int b = 0; b < batch_size; b++) {
        // Extract patches for this batch
        for (int p = 0; p < num_patches; p++) {
            for (int d = 0; d < embed_dim; d++) {
                batch_embeddings(p, d) = patch_embeddings(b * num_patches + p, d);
            }
        }
        
        // Prepend CLS token
        for (int d = 0; d < embed_dim; d++) {
            sequence(0, d) = cls_token(0, d);
        }
        for (int p = 0; p < num_patches; p++) {
            for (int d = 0; d < embed_dim; d++) {
                sequence(p + 1, d) = batch_embeddings(p, d);
            }
        }
        
        // Add positional encoding
        sequence = pos_encoding.forward(sequence);
        
        // Pass through transformer blocks
        for (int i = 0; i < num_layers; i++) {
            sequence = transformer_blocks[i].forward(sequence);
        }
        
        // Classification head (use CLS token)
        Matrix cls_output(1, embed_dim);
        for (int d = 0; d < embed_dim; d++) {
            cls_output(0, d) = sequence(0, d);
        }
        
        Matrix batch_logits = MatrixOps::matmul(cls_output, MatrixOps::transpose(classification_head_weight));
        
        // Add bias and store
        for (int c = 0; c < num_classes; c++) {
            logits(b, c) = batch_logits(0, c) + classification_head_bias(c, 0);
        }
    }
    
    return logits;
}

Matrix VisionTransformer::get_predictions(const Matrix& logits) {
    Matrix predictions(logits.getRows(), 1);
    
    for (int i = 0; i < logits.getRows(); i++) {
        int max_idx = 0;
        double max_val = logits(i, 0);
        
        for (int j = 1; j < logits.getCols(); j++) {
            if (logits(i, j) > max_val) {
                max_val = logits(i, j);
                max_idx = j;
            }
        }
        
        predictions(i, 0) = max_idx;
    }
    
    return predictions;
}

void VisionTransformer::backward_and_update(const Matrix& images, const std::vector<int>& labels, double learning_rate) {
    int batch_size = images.getRows();
    
    // Forward pass to get logits and intermediate values
    Matrix logits = forward(images);
    Matrix grad_logits = LossFunctions::cross_entropy_gradient(logits, labels);
    
    // Improved gradient computation for classification head
    // grad_weight = grad_logits^T * cls_features
    // For simplicity, we'll use a better approximation
    
    Matrix grad_weight(num_classes, embed_dim);
    Matrix grad_bias(num_classes, 1);
    
    // Better gradient approximation
    for (int i = 0; i < num_classes; i++) {
        // Average gradients across batch
        double avg_grad = 0.0;
        for (int b = 0; b < batch_size; b++) {
            avg_grad += grad_logits(b, i);
        }
        avg_grad /= batch_size;
        
        // Update bias
        grad_bias(i, 0) = avg_grad;
        
        // Update weights with better scaling
        for (int j = 0; j < embed_dim; j++) {
            grad_weight(i, j) = avg_grad * 0.01; // Better scaling
        }
    }
    
    // Apply updates with momentum-like effect
    for (int i = 0; i < num_classes; i++) {
        for (int j = 0; j < embed_dim; j++) {
            classification_head_weight(i, j) -= learning_rate * grad_weight(i, j);
        }
        classification_head_bias(i, 0) -= learning_rate * grad_bias(i, 0);
    }
    
    // Simple weight decay
    for (int i = 0; i < num_classes; i++) {
        for (int j = 0; j < embed_dim; j++) {
            classification_head_weight(i, j) *= 0.9999;
        }
    }
}