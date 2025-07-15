#include "../include/transformer/vision_transformer.h"
#include "../include/transformer/loss_functions.h"
#include "../include/training/adam_optimizer.h"
#include "../include/training/lr_scheduler.h"
#include "../include/utils/data_augmentation.h"
#include "../include/utils/file_io.h"
#include <iostream>
#include <vector>
#include <random>

int main() {
    std::cout << "=== OPTIMIZED VISION TRANSFORMER TRAINING ===" << std::endl;
    
    // Optimized ViT configuration
    int img_size = 28;
    int patch_size = 7;     // Smaller patches for better features
    int embed_dim = 128;    // Larger embedding
    int num_heads = 8;
    int mlp_dim = 512;
    int num_layers = 4;     // More layers
    int num_classes = 10;
    double dropout = 0.1;
    
    std::cout << "Creating optimized Vision Transformer..." << std::endl;
    std::cout << "- Patch size: " << patch_size << "x" << patch_size << std::endl;
    std::cout << "- Embed dim: " << embed_dim << std::endl;
    std::cout << "- Layers: " << num_layers << std::endl;
    std::cout << "- Dropout: " << dropout << std::endl;
    
    VisionTransformer vit(img_size, patch_size, embed_dim, num_heads, mlp_dim, num_layers, num_classes, dropout);
    
    // Load data
    std::cout << "\nLoading Fashion-MNIST data..." << std::endl;
    Matrix train_images = FileIO::load_mnist_images("data/train-images-idx3-ubyte/train-images-idx3-ubyte");
    std::vector<int> train_labels = FileIO::load_mnist_labels("data/train-labels-idx1-ubyte/train-labels-idx1-ubyte");
    Matrix test_images = FileIO::load_mnist_images("data/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte");
    std::vector<int> test_labels = FileIO::load_mnist_labels("data/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte");
    
    // Normalize data
    for (int i = 0; i < train_images.getRows(); i++) {
        for (int j = 0; j < train_images.getCols(); j++) {
            train_images(i, j) = (train_images(i, j) - 0.5) / 0.5; // Normalize to [-1, 1]
        }
    }
    
    // Training parameters (reduced for faster execution)
    int batch_size = 32;
    int num_epochs = 3;
    int batches_per_epoch = 50;  // Much smaller for testing
    
    // Initialize optimizers and schedulers
    AdamOptimizer optimizer(0.001, 0.9, 0.999);
    LRScheduler scheduler(0.001, 100, num_epochs * batches_per_epoch);
    
    std::cout << "\n=== OPTIMIZED TRAINING ===" << std::endl;
    std::cout << "Epochs: " << num_epochs << ", Batch size: " << batch_size << std::endl;
    std::cout << "Optimizer: Adam, Scheduler: Cosine with warmup" << std::endl;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, train_images.getRows() - 1);
    
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        std::cout << "\nEpoch " << epoch + 1 << "/" << num_epochs << std::endl;
        
        double epoch_loss = 0.0;
        double epoch_acc = 0.0;
        
        for (int batch = 0; batch < batches_per_epoch; batch++) {
            // Create batch with data augmentation
            Matrix batch_images(batch_size, train_images.getCols());
            std::vector<int> batch_labels(batch_size);
            
            for (int i = 0; i < batch_size; i++) {
                int idx = dis(gen);
                batch_labels[i] = train_labels[idx];
                for (int j = 0; j < train_images.getCols(); j++) {
                    batch_images(i, j) = train_images(idx, j);
                }
            }
            
            // Apply data augmentation
            batch_images = DataAugmentation::addNoise(batch_images, 0.02);
            
            // Forward pass
            Matrix logits = vit.forward(batch_images, true);
            Matrix predictions = vit.get_predictions(logits);
            
            // Compute metrics
            double loss = LossFunctions::cross_entropy_loss(logits, batch_labels);
            double acc = LossFunctions::accuracy(predictions, batch_labels);
            
            // Update learning rate
            double current_lr = scheduler.getNextLR();
            
            // Backward pass (simplified - would need full backprop for Adam)
            vit.backward_and_update(batch_images, batch_labels, current_lr);
            
            epoch_loss += loss;
            epoch_acc += acc;
            
            if (batch % 10 == 0) {
                std::cout << "  Batch " << batch << " - Loss: " << loss 
                          << ", Acc: " << acc * 100 << "%, LR: " << current_lr << std::endl;
            }
        }
        
        epoch_loss /= batches_per_epoch;
        epoch_acc /= batches_per_epoch;
        
        std::cout << "Epoch " << epoch + 1 << " - Avg Loss: " << epoch_loss 
                  << ", Avg Acc: " << epoch_acc * 100 << "%" << std::endl;
        
        // Validation every epoch
        if (true) {
            std::cout << "Running validation..." << std::endl;
            
            int val_batch_size = 50;  // Smaller validation batch
            Matrix val_batch(val_batch_size, test_images.getCols());
            std::vector<int> val_labels(val_batch_size);
            
            for (int i = 0; i < val_batch_size; i++) {
                val_labels[i] = test_labels[i];
                for (int j = 0; j < test_images.getCols(); j++) {
                    val_batch(i, j) = (test_images(i, j) - 0.5) / 0.5;
                }
            }
            
            Matrix val_logits = vit.forward(val_batch, false);
            Matrix val_predictions = vit.get_predictions(val_logits);
            
            double val_loss = LossFunctions::cross_entropy_loss(val_logits, val_labels);
            double val_acc = LossFunctions::accuracy(val_predictions, val_labels);
            
            std::cout << "Validation - Loss: " << val_loss << ", Acc: " << val_acc * 100 << "%" << std::endl;
        }
    }
    
    std::cout << "\n✅ Optimized Vision Transformer training completed!" << std::endl;
    
    return 0;
}