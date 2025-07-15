#pragma once
#include "../matrix/matrix.h"
#include <vector>

class LossFunctions {
public:
    static double cross_entropy_loss(const Matrix& logits, const std::vector<int>& labels);
    static Matrix softmax(const Matrix& logits);
    static double accuracy(const Matrix& predictions, const std::vector<int>& labels);
    static Matrix cross_entropy_gradient(const Matrix& logits, const std::vector<int>& labels);
};