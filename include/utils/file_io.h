#ifndef FILE_IO_H
#define FILE_IO_H

#include <string>
#include <vector>
#include "../matrix/matrix.h"

namespace FileIO {
    // Guardar matriz en archivo
    bool saveMatrix(const Matrix& matrix, const std::string& filename);
    
    // Cargar matriz desde archivo
    Matrix loadMatrix(const std::string& filename);
    
    // Guardar vector en archivo
    bool saveVector(const std::vector<double>& vec, const std::string& filename);
    
    // Cargar vector desde archivo
    std::vector<double> loadVector(const std::string& filename);
    
    // Cargar imágenes MNIST
    Matrix load_mnist_images(const std::string& filename);
    
    // Cargar etiquetas MNIST
    std::vector<int> load_mnist_labels(const std::string& filename);
}

#endif // FILE_IO_H