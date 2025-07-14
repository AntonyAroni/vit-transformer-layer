#include "../../include/utils/file_io.h"
#include <fstream>
#include <iostream>

namespace FileIO {
    bool saveMatrix(const Matrix& matrix, const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) return false;
        
        file << matrix.getRows() << " " << matrix.getCols() << "\n";
        for (int i = 0; i < matrix.getRows(); ++i) {
            for (int j = 0; j < matrix.getCols(); ++j) {
                file << matrix(i, j) << " ";
            }
            file << "\n";
        }
        return true;
    }
    
    Matrix loadMatrix(const std::string& filename) {
        std::ifstream file(filename);
        int rows, cols;
        file >> rows >> cols;
        
        Matrix matrix(rows, cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                double val;
                file >> val;
                matrix(i, j) = val;
            }
        }
        return matrix;
    }
    
    bool saveVector(const std::vector<double>& vec, const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) return false;
        
        file << vec.size() << "\n";
        for (double val : vec) {
            file << val << " ";
        }
        return true;
    }
    
    std::vector<double> loadVector(const std::string& filename) {
        std::ifstream file(filename);
        int size;
        file >> size;
        
        std::vector<double> vec(size);
        for (int i = 0; i < size; ++i) {
            file >> vec[i];
        }
        return vec;
    }
    
    Matrix load_mnist_images(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) throw std::runtime_error("Cannot open file");
        
        int magic, num_images, rows, cols;
        file.read(reinterpret_cast<char*>(&magic), 4);
        file.read(reinterpret_cast<char*>(&num_images), 4);
        file.read(reinterpret_cast<char*>(&rows), 4);
        file.read(reinterpret_cast<char*>(&cols), 4);
        
        // Convert from big-endian
        magic = __builtin_bswap32(magic);
        num_images = __builtin_bswap32(num_images);
        rows = __builtin_bswap32(rows);
        cols = __builtin_bswap32(cols);
        
        Matrix images(num_images, rows * cols);
        for (int i = 0; i < num_images; ++i) {
            for (int j = 0; j < rows * cols; ++j) {
                unsigned char pixel;
                file.read(reinterpret_cast<char*>(&pixel), 1);
                images(i, j) = static_cast<double>(pixel) / 255.0;
            }
        }
        return images;
    }
    
    std::vector<int> load_mnist_labels(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) throw std::runtime_error("Cannot open file");
        
        int magic, num_labels;
        file.read(reinterpret_cast<char*>(&magic), 4);
        file.read(reinterpret_cast<char*>(&num_labels), 4);
        
        // Convert from big-endian
        magic = __builtin_bswap32(magic);
        num_labels = __builtin_bswap32(num_labels);
        
        std::vector<int> labels(num_labels);
        for (int i = 0; i < num_labels; ++i) {
            unsigned char label;
            file.read(reinterpret_cast<char*>(&label), 1);
            labels[i] = static_cast<int>(label);
        }
        return labels;
    }
}