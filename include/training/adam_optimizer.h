#pragma once
#include "../matrix/matrix.h"
#include <unordered_map>

class AdamOptimizer {
private:
    double learning_rate;
    double beta1, beta2;
    double epsilon;
    int t; // time step
    std::unordered_map<void*, Matrix> m_cache; // momentum
    std::unordered_map<void*, Matrix> v_cache; // velocity

public:
    AdamOptimizer(double lr = 0.001, double b1 = 0.9, double b2 = 0.999, double eps = 1e-8);
    void update(Matrix& weights, const Matrix& gradients);
    void reset();
};