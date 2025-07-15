#include "../../include/utils/data_augmentation.h"
#include <random>
#include <cmath>

Matrix DataAugmentation::addNoise(const Matrix& image, double noise_level) {
    Matrix result = image;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> noise(0.0, noise_level);
    
    for (int i = 0; i < result.getRows(); i++) {
        for (int j = 0; j < result.getCols(); j++) {
            result(i, j) = std::max(0.0, std::min(1.0, result(i, j) + noise(gen)));
        }
    }
    return result;
}

Matrix DataAugmentation::normalize(const Matrix& image, double mean, double std) {
    Matrix result = image;
    for (int i = 0; i < result.getRows(); i++) {
        for (int j = 0; j < result.getCols(); j++) {
            result(i, j) = (result(i, j) - mean) / std;
        }
    }
    return result;
}

Matrix DataAugmentation::randomCrop(const Matrix& image, int crop_size) {
    // Simple implementation - just return original for now
    return image;
}

std::vector<Matrix> DataAugmentation::augmentBatch(const std::vector<Matrix>& batch) {
    std::vector<Matrix> augmented;
    for (const auto& img : batch) {
        Matrix aug = addNoise(img, 0.05);
        aug = normalize(aug);
        augmented.push_back(aug);
    }
    return augmented;
}