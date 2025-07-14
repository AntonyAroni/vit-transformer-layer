#include "../include/utils/file_io.h"
#include "../include/matrix/matrix.h"
#include <iostream>
#include <vector>
#include <string>

// Fashion-MNIST class labels
std::vector<std::string> fashion_labels = {
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
};


/*
 g++ -std=c++17 -I. tests/02_fashion_mnist_example.cpp src/matrix/matrix.cpp src/utils/file_io.cpp -o fashion_test && ./fashion_test
*/


int main() {
    try {
        std::cout << "Loading Fashion-MNIST dataset..." << std::endl;

        // Load training data
        Matrix train_images = FileIO::load_mnist_images("data/train-images-idx3-ubyte/train-images-idx3-ubyte");
        std::vector<int> train_labels = FileIO::load_mnist_labels("data/train-labels-idx1-ubyte/train-labels-idx1-ubyte");
        
        std::cout << "Training images: " << train_images.getRows() << " x " << train_images.getCols() << std::endl;
        std::cout << "Training labels: " << train_labels.size() << std::endl;
        
        // Load test data
        Matrix test_images = FileIO::load_mnist_images("data/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte");
        std::vector<int> test_labels = FileIO::load_mnist_labels("data/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte");
        
        std::cout << "Test images: " << test_images.getRows() << " x " << test_images.getCols() << std::endl;
        std::cout << "Test labels: " << test_labels.size() << std::endl;
        
        // Show first 10 samples with Fashion-MNIST labels
        std::cout << "\nFirst 10 Fashion-MNIST samples:" << std::endl;
        std::cout << "===============================" << std::endl;
        
        for (int i = 0; i < 10; ++i) {
            std::cout << "Sample " << i << ": " << fashion_labels[train_labels[i]] 
                      << " (class " << train_labels[i] << ")" << std::endl;
        }
        
        // Count class distribution
        std::vector<int> class_counts(10, 0);
        for (int label : train_labels) {
            class_counts[label]++;
        }
        
        std::cout << "\nClass distribution in training set:" << std::endl;
        std::cout << "===================================" << std::endl;
        for (int i = 0; i < 10; ++i) {
            std::cout << i << ". " << fashion_labels[i] << ": " << class_counts[i] << " samples" << std::endl;
        }
        
        std::cout << "\n✅ Fashion-MNIST data loaded successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}