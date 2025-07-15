#include "../../include/training/adam_optimizer.h"
#include <cmath>

AdamOptimizer::AdamOptimizer(double lr, double b1, double b2, double eps) 
    : learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps), t(0) {}

void AdamOptimizer::update(Matrix& weights, const Matrix& gradients) {
    t++;
    void* key = &weights;
    
    // Initialize momentum and velocity if not exists
    if (m_cache.find(key) == m_cache.end()) {
        m_cache[key] = Matrix::zeros(weights.getRows(), weights.getCols());
        v_cache[key] = Matrix::zeros(weights.getRows(), weights.getCols());
    }
    
    Matrix& m = m_cache[key];
    Matrix& v = v_cache[key];
    
    // Update biased first and second moment estimates
    for (int i = 0; i < weights.getRows(); i++) {
        for (int j = 0; j < weights.getCols(); j++) {
            double grad = gradients(i, j);
            
            // Update momentum
            m(i, j) = beta1 * m(i, j) + (1 - beta1) * grad;
            
            // Update velocity
            v(i, j) = beta2 * v(i, j) + (1 - beta2) * grad * grad;
            
            // Bias correction
            double m_hat = m(i, j) / (1 - pow(beta1, t));
            double v_hat = v(i, j) / (1 - pow(beta2, t));
            
            // Update weights
            weights(i, j) -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
        }
    }
}

void AdamOptimizer::reset() {
    t = 0;
    m_cache.clear();
    v_cache.clear();
}