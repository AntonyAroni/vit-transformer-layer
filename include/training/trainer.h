#ifndef TRAINER_H
#define TRAINER_H

#include "../matrix/matrix.h"
#include "../transformer/transformer_block.h"
#include <vector>

class SimpleClassifier {
private:
    Matrix input_projection;
    TransformerBlock transformer;
    Matrix classifier_weights;
    Matrix classifier_bias;
    size_t input_dim;
    size_t embed_dim;
    size_t num_classes;
    
public:
    SimpleClassifier(size_t input_dim, size_t embed_dim, size_t num_heads, size_t mlp_hidden_dim, size_t num_classes);
    Matrix forward(const Matrix& input);
    double compute_loss(const Matrix& predictions, const std::vector<int>& labels);
    double compute_accuracy(const Matrix& predictions, const std::vector<int>& test_labels);
    void train_step(const Matrix& batch_images, const std::vector<int>& batch_labels, double learning_rate);
};

class Trainer {
public:
    static void train_model(SimpleClassifier& model, 
                          const Matrix& train_images, const std::vector<int>& train_labels,
                          const Matrix& test_images, const std::vector<int>& test_labels,
                          int epochs, int batch_size, double learning_rate);
};

#endif