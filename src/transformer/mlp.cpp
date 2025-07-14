#include "../../include/transformer/mlp.h"
#include "../../include/matrix/matrix_ops.h"
#include "../../include/matrix/activation_functions.h"
#include <cmath>

/*
g++ -std=c++17 -I. test_code/03_test_mlp.cpp src/matrix/matrix.cpp src/matrix/matrix_ops.cpp src/matrix/activation_functions.h.cpp src/transformer/mlp.cpp -o test_mlp && ./test_mlp
 */

MLP::MLP(size_t input_dim, size_t hidden_dim) 
    : input_dim(input_dim), hidden_dim(hidden_dim) {
    initialize_weights();
}

void MLP::initialize_weights() {
    // Xavier initialization
    double scale1 = sqrt(2.0 / input_dim);
    double scale2 = sqrt(2.0 / hidden_dim);
    
    W1 = Matrix::random(input_dim, hidden_dim) * scale1;
    b1 = Matrix::zeros(1, hidden_dim);
    
    W2 = Matrix::random(hidden_dim, input_dim) * scale2;
    b2 = Matrix::zeros(1, input_dim);
}

Matrix MLP::forward(const Matrix& input) {
    // First linear layer: input -> hidden
    Matrix hidden = MatrixOps::matmul(input, W1);
    
    // Add bias (broadcast)
    for (size_t i = 0; i < hidden.getRows(); ++i) {
        for (size_t j = 0; j < hidden.getCols(); ++j) {
            hidden(i, j) += b1(0, j);
        }
    }
    
    // GELU activation
    hidden = ActivationFunctions::gelu(hidden);
    
    // Second linear layer: hidden -> output
    Matrix output = MatrixOps::matmul(hidden, W2);
    
    // Add bias (broadcast)
    for (size_t i = 0; i < output.getRows(); ++i) {
        for (size_t j = 0; j < output.getCols(); ++j) {
            output(i, j) += b2(0, j);
        }
    }
    
    return output;
}