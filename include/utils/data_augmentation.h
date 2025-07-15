#pragma once
#include "../matrix/matrix.h"
#include <vector>

class DataAugmentation {
public:
    static Matrix addNoise(const Matrix& image, double noise_level = 0.1);
    static Matrix randomCrop(const Matrix& image, int crop_size = 24);
    static Matrix normalize(const Matrix& image, double mean = 0.5, double std = 0.5);
    static std::vector<Matrix> augmentBatch(const std::vector<Matrix>& batch);
};