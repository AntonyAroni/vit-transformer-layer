#include "../include/transformer/vision_transformer.h"
#include "../include/utils/file_io.h"
#include <iostream>
#include <vector>

int main() {
    std::cout << "=== COMPLETE VISION TRANSFORMER TEST ===" << std::endl;
    
    // ViT configuration
    int img_size = 28;      // Fashion-MNIST image size
    int patch_size = 7;     // 4x4 patches for 28x28 images
    int embed_dim = 256;
    int num_heads = 8;
    int mlp_dim = 1024;
    int num_layers = 6;
    int num_classes = 10;   // Fashion-MNIST classes
    
    std::cout << "Creating Vision Transformer..." << std::endl;
    std::cout << "- Image size: " << img_size << "x" << img_size << std::endl;
    std::cout << "- Patch size: " << patch_size << "x" << patch_size << std::endl;
    std::cout << "- Embed dim: " << embed_dim << std::endl;
    std::cout << "- Num heads: " << num_heads << std::endl;
    std::cout << "- Num layers: " << num_layers << std::endl;
    std::cout << "- Num classes: " << num_classes << std::endl;
    
    VisionTransformer vit(img_size, patch_size, embed_dim, num_heads, mlp_dim, num_layers, num_classes);
    
    // Load Fashion-MNIST test data
    std::cout << "\nLoading Fashion-MNIST data..." << std::endl;
    Matrix test_images = FileIO::load_mnist_images("data/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte");
    std::vector<int> test_labels = FileIO::load_mnist_labels("data/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte");
    
    std::cout << "Test images shape: " << test_images.getRows() << " x " << test_images.getCols() << std::endl;
    std::cout << "Test labels: " << test_labels.size() << std::endl;
    
    // Test with small batch
    int batch_size = 5;
    Matrix batch_images(batch_size, test_images.getCols());
    
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < test_images.getCols(); j++) {
            batch_images(i, j) = test_images(i, j);
        }
    }
    
    std::cout << "\nRunning forward pass..." << std::endl;
    Matrix logits = vit.forward(batch_images);
    Matrix predictions = vit.get_predictions(logits);
    
    std::cout << "Logits shape: " << logits.getRows() << " x " << logits.getCols() << std::endl;
    std::cout << "Predictions shape: " << predictions.getRows() << " x " << predictions.getCols() << std::endl;
    
    // Show results
    std::vector<std::string> class_names = {
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    };
    
    std::cout << "\nPredictions vs Ground Truth:" << std::endl;
    for (int i = 0; i < batch_size; i++) {
        int pred = (int)predictions(i, 0);
        int true_label = test_labels[i];
        std::cout << "Sample " << i << ": Predicted=" << class_names[pred] 
                  << ", True=" << class_names[true_label] << std::endl;
    }
    
    std::cout << "\n✅ Complete Vision Transformer test completed!" << std::endl;
    
    return 0;
}