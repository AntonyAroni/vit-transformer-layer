#include "../include/transformer/vision_transformer.h"
#include "../include/transformer/loss_functions.h"
#include "../include/utils/file_io.h"
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>

int main() {
    std::cout << "=== VISION TRANSFORMER TRAINING DEMO ===" << std::endl;
    
    // ViT configuration
    int img_size = 28;
    int patch_size = 7;
    int embed_dim = 128;    // Smaller for demo
    int num_heads = 4;
    int mlp_dim = 512;
    int num_layers = 3;     // Fewer layers for demo
    int num_classes = 10;
    
    std::cout << "Creating Vision Transformer..." << std::endl;
    std::cout << "- Embed dim: " << embed_dim << std::endl;
    std::cout << "- Num heads: " << num_heads << std::endl;
    std::cout << "- Num layers: " << num_layers << std::endl;
    
    VisionTransformer vit(img_size, patch_size, embed_dim, num_heads, mlp_dim, num_layers, num_classes);
    
    // Load Fashion-MNIST data
    std::cout << "\nLoading Fashion-MNIST data..." << std::endl;
    Matrix train_images = FileIO::load_mnist_images("data/train-images-idx3-ubyte/train-images-idx3-ubyte");
    std::vector<int> train_labels = FileIO::load_mnist_labels("data/train-labels-idx1-ubyte/train-labels-idx1-ubyte");
    Matrix test_images = FileIO::load_mnist_images("data/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte");
    std::vector<int> test_labels = FileIO::load_mnist_labels("data/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte");
    
    std::cout << "Train images: " << train_images.getRows() << " x " << train_images.getCols() << std::endl;
    std::cout << "Test images: " << test_images.getRows() << " x " << test_images.getCols() << std::endl;
    
    // Training simulation (forward pass only)
    std::cout << "\n=== TRAINING SIMULATION ===" << std::endl;
    
    int batch_size = 32;
    int num_batches = 5; // Just a few batches for demo
    
    std::random_device rd;
    std::mt19937 gen(rd());
    
    for (int batch = 0; batch < num_batches; batch++) {
        // Create random batch
        Matrix batch_images(batch_size, train_images.getCols());
        std::vector<int> batch_labels(batch_size);
        
        std::uniform_int_distribution<> dis(0, train_images.getRows() - 1);
        
        for (int i = 0; i < batch_size; i++) {
            int idx = dis(gen);
            batch_labels[i] = train_labels[idx];
            for (int j = 0; j < train_images.getCols(); j++) {
                batch_images(i, j) = train_images(idx, j);
            }
        }
        
        // Forward pass
        Matrix logits = vit.forward(batch_images);
        Matrix predictions = vit.get_predictions(logits);
        
        // Compute metrics
        double loss = LossFunctions::cross_entropy_loss(logits, batch_labels);
        double acc = LossFunctions::accuracy(predictions, batch_labels);
        
        std::cout << "Batch " << batch + 1 << "/" << num_batches 
                  << " - Loss: " << loss << ", Accuracy: " << acc * 100 << "%" << std::endl;
    }
    
    // Test evaluation
    std::cout << "\n=== TEST EVALUATION ===" << std::endl;
    
    int test_batch_size = 100;
    Matrix test_batch(test_batch_size, test_images.getCols());
    std::vector<int> test_batch_labels(test_batch_size);
    
    for (int i = 0; i < test_batch_size; i++) {
        test_batch_labels[i] = test_labels[i];
        for (int j = 0; j < test_images.getCols(); j++) {
            test_batch(i, j) = test_images(i, j);
        }
    }
    
    Matrix test_logits = vit.forward(test_batch);
    Matrix test_predictions = vit.get_predictions(test_logits);
    
    double test_loss = LossFunctions::cross_entropy_loss(test_logits, test_batch_labels);
    double test_acc = LossFunctions::accuracy(test_predictions, test_batch_labels);
    
    std::cout << "Test Loss: " << test_loss << std::endl;
    std::cout << "Test Accuracy: " << test_acc * 100 << "%" << std::endl;
    
    // Show some predictions
    std::vector<std::string> class_names = {
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    };
    
    std::cout << "\nSample predictions:" << std::endl;
    for (int i = 0; i < 10; i++) {
        int pred = (int)test_predictions(i, 0);
        int true_label = test_batch_labels[i];
        std::string status = (pred == true_label) ? "✓" : "✗";
        std::cout << status << " Sample " << i << ": " << class_names[pred] 
                  << " (true: " << class_names[true_label] << ")" << std::endl;
    }
    
    std::cout << "\n✅ Vision Transformer training demo completed!" << std::endl;
    
    return 0;
}