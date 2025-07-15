#include "../../include/training/optimizer.h"

void SGDOptimizer::update(Matrix& weights, const Matrix& gradients) {
    for (int i = 0; i < weights.getRows(); i++) {
        for (int j = 0; j < weights.getCols(); j++) {
            weights(i, j) -= learning_rate * gradients(i, j);
        }
    }
}