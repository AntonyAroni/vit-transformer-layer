#include "../include/utils/file_io.h"
#include "../include/matrix/matrix.h"
#include "../include/matrix/activation_functions.h"
#include <iostream>
#include <vector>
#include <string>

std::vector<std::string> fashion_labels = {
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
};

/*
 g++ -std=c++17 -I. tests/03_test_fashion_vit.cpp src/matrix/matrix.cpp src/utils/file_io.cpp -o fashion_test_vit && ./fashion_test_vit
*/
int main() {
    try {
        std::cout << "Testing Fashion-MNIST data loading..." << std::endl;
        
        std::cout << "Loading Fashion-MNIST data..." << std::endl;
        Matrix test_images = FileIO::load_mnist_images("data/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte");
        std::vector<int> test_labels = FileIO::load_mnist_labels("data/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte");
        
        std::cout << "Fashion-MNIST test set: " << test_images.getRows() << " images" << std::endl;
        std::cout << "Image dimensions: " << test_images.getCols() << " pixels" << std::endl;
        
        std::cout << "\nFirst 10 Fashion-MNIST samples:" << std::endl;
        std::cout << "==============================" << std::endl;
        
        for (size_t i = 0; i < 10; ++i) {
            std::cout << "Image " << i << ": " << fashion_labels[test_labels[i]] 
                      << " (label: " << test_labels[i] << ")" << std::endl;
        }
        
        std::cout << "\n✅ Fashion-MNIST data loaded successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}