#include "../include/transformer/vision_transformer.h"
#include "../include/transformer/loss_functions.h"
#include "../include/utils/file_io.h"
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>

int main() {
    std::cout << "=== VISION TRANSFORMER TRAINING ===" << std::endl;
    
    // Smaller ViT for faster training
    int img_size = 28;
    int patch_size = 14;    // Larger patches = fewer tokens
    int embed_dim = 64;     // Smaller embedding
    int num_heads = 4;
    int mlp_dim = 128;
    int num_layers = 2;     // Fewer layers
    int num_classes = 10;
    
    std::cout << "Creating compact Vision Transformer..." << std::endl;
    std::cout << "- Patch size: " << patch_size << "x" << patch_size << std::endl;
    std::cout << "- Embed dim: " << embed_dim << std::endl;
    std::cout << "- Layers: " << num_layers << std::endl;
    
    VisionTransformer vit(img_size, patch_size, embed_dim, num_heads, mlp_dim, num_layers, num_classes);
    
    // Load data
    std::cout << "\nLoading Fashion-MNIST data..." << std::endl;
    Matrix train_images = FileIO::load_mnist_images("data/train-images-idx3-ubyte/train-images-idx3-ubyte");
    std::vector<int> train_labels = FileIO::load_mnist_labels("data/train-labels-idx1-ubyte/train-labels-idx1-ubyte");
    Matrix test_images = FileIO::load_mnist_images("data/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte");
    std::vector<int> test_labels = FileIO::load_mnist_labels("data/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte");
    
    // Training parameters
    int batch_size = 32;
    int num_epochs = 5;
    double learning_rate = 0.1;  // Higher learning rate
    int batches_per_epoch = 200; // More batches
    
    std::cout << "\n=== TRAINING ===" << std::endl;
    std::cout << "Epochs: " << num_epochs << ", Batch size: " << batch_size << std::endl;
    std::cout << "Learning rate: " << learning_rate << std::endl;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, train_images.getRows() - 1);
    
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        std::cout << "\nEpoch " << epoch + 1 << "/" << num_epochs << std::endl;
        
        double epoch_loss = 0.0;
        double epoch_acc = 0.0;
        
        for (int batch = 0; batch < batches_per_epoch; batch++) {
            // Create batch
            Matrix batch_images(batch_size, train_images.getCols());
            std::vector<int> batch_labels(batch_size);
            
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
            
            // Backward pass and update
            vit.backward_and_update(batch_images, batch_labels, learning_rate);
            
            epoch_loss += loss;
            epoch_acc += acc;
            
            if (batch % 20 == 0) {
                std::cout << "  Batch " << batch << " - Loss: " << loss 
                          << ", Acc: " << acc * 100 << "%" << std::endl;
            }
        }
        
        epoch_loss /= batches_per_epoch;
        epoch_acc /= batches_per_epoch;
        
        std::cout << "Epoch " << epoch + 1 << " - Avg Loss: " << epoch_loss 
                  << ", Avg Acc: " << epoch_acc * 100 << "%" << std::endl;
    }
    
    // Final test evaluation
    std::cout << "\n=== FINAL TEST EVALUATION ===" << std::endl;
    
    int test_batch_size = 100;
    Matrix test_batch(test_batch_size, test_images.getCols());
    std::vector<int> test_batch_labels(test_batch_size);
    
    for (int i = 0; i < test_batch_size; i++) {
        test_batch_labels[i] = test_labels[i];
        for (int j = 0; j < test_images.getCols(); j++) {
            test_batch(i, j) = test_images(i, j);
        }
    }
    
    Matrix final_logits = vit.forward(test_batch);
    Matrix final_predictions = vit.get_predictions(final_logits);
    
    double final_loss = LossFunctions::cross_entropy_loss(final_logits, test_batch_labels);
    double final_acc = LossFunctions::accuracy(final_predictions, test_batch_labels);
    
    std::cout << "Final Test Loss: " << final_loss << std::endl;
    std::cout << "Final Test Accuracy: " << final_acc * 100 << "%" << std::endl;
    
    std::cout << "\n✅ Vision Transformer training completed!" << std::endl;
    
    return 0;
}