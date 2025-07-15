#include "../../include/transformer/loss_functions.h"
#include <cmath>
#include <algorithm>

double LossFunctions::cross_entropy_loss(const Matrix& logits, const std::vector<int>& labels) {
    Matrix probs = softmax(logits);
    double total_loss = 0.0;
    
    for (int i = 0; i < logits.getRows(); i++) {
        int true_label = labels[i];
        double prob = std::max(probs(i, true_label), 1e-15); // Avoid log(0)
        total_loss -= log(prob);
    }
    
    return total_loss / logits.getRows();
}

Matrix LossFunctions::softmax(const Matrix& logits) {
    Matrix result(logits.getRows(), logits.getCols());
    
    for (int i = 0; i < logits.getRows(); i++) {
        // Find max for numerical stability
        double max_val = logits(i, 0);
        for (int j = 1; j < logits.getCols(); j++) {
            max_val = std::max(max_val, logits(i, j));
        }
        
        // Compute exp and sum
        double sum_exp = 0.0;
        for (int j = 0; j < logits.getCols(); j++) {
            double exp_val = exp(logits(i, j) - max_val);
            result(i, j) = exp_val;
            sum_exp += exp_val;
        }
        
        // Normalize
        for (int j = 0; j < logits.getCols(); j++) {
            result(i, j) /= sum_exp;
        }
    }
    
    return result;
}

double LossFunctions::accuracy(const Matrix& predictions, const std::vector<int>& labels) {
    int correct = 0;
    
    for (int i = 0; i < predictions.getRows(); i++) {
        int pred = (int)predictions(i, 0);
        if (pred == labels[i]) {
            correct++;
        }
    }
    
    return (double)correct / predictions.getRows();
}

Matrix LossFunctions::cross_entropy_gradient(const Matrix& logits, const std::vector<int>& labels) {
    Matrix probs = softmax(logits);
    Matrix grad(logits.getRows(), logits.getCols());
    
    for (int i = 0; i < logits.getRows(); i++) {
        for (int j = 0; j < logits.getCols(); j++) {
            grad(i, j) = probs(i, j);
            if (j == labels[i]) {
                grad(i, j) -= 1.0;
            }
        }
    }
    
    // Average over batch
    for (int i = 0; i < grad.getRows(); i++) {
        for (int j = 0; j < grad.getCols(); j++) {
            grad(i, j) /= logits.getRows();
        }
    }
    
    return grad;
}