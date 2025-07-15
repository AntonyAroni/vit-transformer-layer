#include "../../include/training/trainer.h"
#include "../../include/matrix/matrix_ops.h"
#include "../../include/matrix/activation_functions.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <random>

SimpleClassifier::SimpleClassifier(size_t input_dim, size_t embed_dim, size_t num_heads, size_t mlp_hidden_dim, size_t num_classes)
    : transformer(embed_dim, num_heads, mlp_hidden_dim), input_dim(input_dim), embed_dim(embed_dim), num_classes(num_classes) {
    
    input_projection = Matrix::random(input_dim, embed_dim, -0.1, 0.1);
    classifier_weights = Matrix::random(embed_dim, num_classes, -0.1, 0.1);
    classifier_bias = Matrix::zeros(1, num_classes);
}

Matrix SimpleClassifier::forward(const Matrix& input) {
    // Project input to embedding dimension
    Matrix projected = MatrixOps::matmul(input, input_projection);
    Matrix transformer_output = transformer.forward(projected);
    
    // Global average pooling
    Matrix pooled = Matrix::zeros(input.getRows(), embed_dim);
    for (size_t i = 0; i < input.getRows(); ++i) {
        for (size_t j = 0; j < embed_dim; ++j) {
            pooled(i, j) = transformer_output(i, j);
        }
    }
    
    // Classification layer
    Matrix logits = MatrixOps::matmul(pooled, classifier_weights);
    for (size_t i = 0; i < input.getRows(); ++i) {
        for (size_t j = 0; j < num_classes; ++j) {
            logits(i, j) += classifier_bias(0, j);
        }
    }
    
    return ActivationFunctions::softmax(logits);
}

double SimpleClassifier::compute_loss(const Matrix& predictions, const std::vector<int>& labels) {
    double loss = 0.0;
    for (size_t i = 0; i < labels.size(); ++i) {
        loss -= std::log(std::max(predictions(i, labels[i]), 1e-15));
    }
    return loss / labels.size();
}

double SimpleClassifier::compute_accuracy(const Matrix& predictions, const std::vector<int>& test_labels) {
    int correct = 0;
    for (size_t i = 0; i < test_labels.size(); ++i) {
        int predicted_class = 0;
        double max_prob = predictions(i, 0);
        for (size_t j = 1; j < num_classes; ++j) {
            if (predictions(i, j) > max_prob) {
                max_prob = predictions(i, j);
                predicted_class = j;
            }
        }
        if (predicted_class == test_labels[i]) correct++;
    }
    return static_cast<double>(correct) / test_labels.size();
}

void SimpleClassifier::train_step(const Matrix& batch_images, const std::vector<int>& batch_labels, double learning_rate) {
    Matrix predictions = forward(batch_images);
    
    // Compute gradients for classifier layer
    Matrix grad_weights = Matrix::zeros(embed_dim, num_classes);
    Matrix grad_bias = Matrix::zeros(1, num_classes);
    
    // Project input for gradient computation
    Matrix projected = MatrixOps::matmul(batch_images, input_projection);
    Matrix features = transformer.forward(projected);
    
    // Compute classification gradients
    for (size_t i = 0; i < batch_labels.size(); ++i) {
        for (size_t j = 0; j < num_classes; ++j) {
            double target = (j == batch_labels[i]) ? 1.0 : 0.0;
            double error = predictions(i, j) - target;
            
            // Update weight gradients
            for (size_t k = 0; k < embed_dim; ++k) {
                grad_weights(k, j) += error * features(i, k);
            }
            grad_bias(0, j) += error;
        }
    }
    
    // Apply gradients
    double batch_size = static_cast<double>(batch_labels.size());
    for (size_t i = 0; i < embed_dim; ++i) {
        for (size_t j = 0; j < num_classes; ++j) {
            classifier_weights(i, j) -= learning_rate * grad_weights(i, j) / batch_size;
        }
    }
    for (size_t j = 0; j < num_classes; ++j) {
        classifier_bias(0, j) -= learning_rate * grad_bias(0, j) / batch_size;
    }
}

void Trainer::train_model(SimpleClassifier& model, 
                         const Matrix& train_images, const std::vector<int>& train_labels,
                         const Matrix& test_images, const std::vector<int>& test_labels,
                         int epochs, int batch_size, double learning_rate) {
    
    std::cout << "=== ENTRENAMIENTO TRANSFORMER ===" << std::endl;
    std::cout << "Epochs: " << epochs << ", Batch size: " << batch_size << ", LR: " << learning_rate << std::endl;
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::cout << "\nEpoch " << (epoch + 1) << "/" << epochs << std::endl;
        
        // Training
        int num_batches = std::min(10, static_cast<int>(train_images.getRows()) / batch_size);
        double total_loss = 0.0;
        
        for (int batch = 0; batch < num_batches; ++batch) {
            int start_idx = batch * batch_size;
            int end_idx = std::min(start_idx + batch_size, static_cast<int>(train_images.getRows()));
            
            // Create batch
            Matrix batch_images(end_idx - start_idx, train_images.getCols());
            std::vector<int> batch_labels;
            
            for (int i = start_idx; i < end_idx; ++i) {
                for (size_t j = 0; j < train_images.getCols(); ++j) {
                    batch_images(i - start_idx, j) = train_images(i, j);
                }
                batch_labels.push_back(train_labels[i]);
            }
            
            // Forward pass and loss
            Matrix predictions = model.forward(batch_images);
            double loss = model.compute_loss(predictions, batch_labels);
            total_loss += loss;
            
            // Training step
            model.train_step(batch_images, batch_labels, learning_rate);
            
            std::cout << "  Batch " << (batch + 1) << "/" << num_batches << " - Loss: " << loss << std::endl;
        }
        
        // Test accuracy
        Matrix test_batch(std::min(100, static_cast<int>(test_images.getRows())), test_images.getCols());
        std::vector<int> test_batch_labels;
        
        for (int i = 0; i < std::min(100, static_cast<int>(test_images.getRows())); ++i) {
            for (size_t j = 0; j < test_images.getCols(); ++j) {
                test_batch(i, j) = test_images(i, j);
            }
            test_batch_labels.push_back(test_labels[i]);
        }
        
        Matrix test_predictions = model.forward(test_batch);
        double accuracy = model.compute_accuracy(test_predictions, test_batch_labels);
        
        std::cout << "Epoch " << (epoch + 1) << " - Avg Loss: " << (total_loss / num_batches) 
                  << ", Test Accuracy: " << (accuracy * 100) << "%" << std::endl;
    }
    
    std::cout << "\n✅ Entrenamiento completado!" << std::endl;
}