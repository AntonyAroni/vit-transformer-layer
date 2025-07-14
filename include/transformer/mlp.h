#ifndef MLP_H
#define MLP_H

#include "../matrix/matrix.h"

class MLP {
private:
    size_t input_dim;
    size_t hidden_dim;
    
    Matrix W1, b1;  // First linear layer
    Matrix W2, b2;  // Second linear layer
    
public:
    MLP(size_t input_dim, size_t hidden_dim);
    
    Matrix forward(const Matrix& input);
    void initialize_weights();
};

#endif