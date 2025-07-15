#pragma once
#include "../matrix/matrix.h"

class SGDOptimizer {
private:
    double learning_rate;

public:
    SGDOptimizer(double lr = 0.001) : learning_rate(lr) {}
    void update(Matrix& weights, const Matrix& gradients);
};