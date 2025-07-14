#include "../include/transformer/transformer_block.h"
#include "../include/matrix/matrix.h"
#include <iostream>


/*
g++ -std=c++17 -I. tests/01_test_transformer_layer.cpp src/matrix/matrix.cpp src/matrix/matrix_ops.cpp src/matrix/activation_functions.h.cpp src/transformer/multi_head_attention.cpp src/transformer/mlp.cpp src/transformer/layer_norm.cpp src/transformer/transformer_block.cpp src/utils/file_io.cpp -o test_transformer && ./test_transformer

*/

int main() {
    try {
        std::cout << "=== TRANSFORMER LAYER TEST ===" << std::endl;
        
        // Configuración típica
        size_t embed_dim = 256;
        size_t num_heads = 8;
        size_t mlp_hidden_dim = 1024;
        size_t seq_len = 50;
        
        std::cout << "Creando Transformer Block..." << std::endl;
        std::cout << "- Embed dim: " << embed_dim << std::endl;
        std::cout << "- Num heads: " << num_heads << std::endl;
        std::cout << "- MLP hidden: " << mlp_hidden_dim << std::endl;
        
        TransformerBlock transformer(embed_dim, num_heads, mlp_hidden_dim);
        
        // Input de prueba
        Matrix input = Matrix::random(seq_len, embed_dim);
        std::cout << "\nInput shape: " << input.getRows() << " x " << input.getCols() << std::endl;
        
        // Forward pass
        std::cout << "Ejecutando forward pass..." << std::endl;
        Matrix output = transformer.forward(input);
        
        std::cout << "Output shape: " << output.getRows() << " x " << output.getCols() << std::endl;
        
        // Verificar que las dimensiones son correctas
        if (output.getRows() == input.getRows() && output.getCols() == input.getCols()) {
            std::cout << "✅ Transformer Layer funcionando correctamente!" << std::endl;
        } else {
            std::cout << "❌ Error en dimensiones de salida" << std::endl;
        }
        
        // Mostrar rango de valores
        double min_val = output(0, 0), max_val = output(0, 0);
        for (size_t i = 0; i < output.getRows(); ++i) {
            for (size_t j = 0; j < output.getCols(); ++j) {
                if (output(i, j) < min_val) min_val = output(i, j);
                if (output(i, j) > max_val) max_val = output(i, j);
            }
        }
        std::cout << "Rango de salida: [" << min_val << ", " << max_val << "]" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}