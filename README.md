# Vision Transformer (ViT) Implementation

## Descripción
Implementación completa de un Vision Transformer en C++ que incluye:

### Componentes Principales:
1. **Multi-Head Self-Attention** - Mecanismo de atención con múltiples cabezas
2. **MLP (Feed-Forward Network)** - Red neuronal feed-forward con activación GELU
3. **Layer Normalization** - Normalización de capas
4. **Transformer Block** - Capa completa con conexiones residuales
5. **Patch Embedding** - Conversión de imágenes a patches
6. **Positional Encoding** - Codificación posicional aprendible
7. **Vision Transformer** - Modelo completo con clasificación
8. **Loss Functions** - Funciones de pérdida y métricas
9. **File I/O Utilities** - Carga de datos MNIST/Fashion-MNIST

### Arquitectura Vision Transformer:
```
Image → Patch Embedding → [CLS] + Patches + Positional Encoding
      → Transformer Blocks (N layers) → Classification Head → Logits

Transformer Block:
Input → LayerNorm → Multi-Head Attention → Add (residual) 
      → LayerNorm → MLP → Add (residual) → Output
```

## Estructura de Archivos:
```
transformer_layer_package/
├── include/
│   ├── matrix/                    # Dependencias de matriz
│   │   ├── matrix.h
│   │   ├── matrix_ops.h
│   │   └── activation_functions.h
│   ├── transformer/               # Componentes transformer
│   │   ├── multi_head_attention.h
│   │   ├── mlp.h
│   │   ├── layer_norm.h
│   │   ├── transformer_block.h
│   │   ├── patch_embedding.h      # ✨ NUEVO
│   │   ├── positional_encoding.h  # ✨ NUEVO
│   │   ├── vision_transformer.h   # ✨ NUEVO
│   │   └── loss_functions.h       # ✨ NUEVO
│   └── utils/                     # Utilidades
│       └── file_io.h
├── src/
│   ├── matrix/                    # Implementaciones de matriz
│   │   ├── matrix.cpp
│   │   ├── matrix_ops.cpp
│   │   └── activation_functions.h.cpp
│   ├── transformer/               # Implementaciones transformer
│   │   ├── multi_head_attention.cpp
│   │   ├── mlp.cpp
│   │   ├── layer_norm.cpp
│   │   ├── transformer_block.cpp
│   │   ├── patch_embedding.cpp    # ✨ NUEVO
│   │   ├── positional_encoding.cpp # ✨ NUEVO
│   │   ├── vision_transformer.cpp # ✨ NUEVO
│   │   └── loss_functions.cpp     # ✨ NUEVO
│   └── utils/                     # Implementaciones utilidades
│       └── file_io.cpp
├── tests/
│   ├── 01_test_transformer_layer.cpp
│   ├── 02_fashion_mnist_example.cpp
│   ├── 03_test_fashion_vit.cpp
│   ├── 04_complete_vit_test.cpp   # ✨ NUEVO
│   └── 05_vit_training_demo.cpp   # ✨ NUEVO
├── data/                          # Datos Fashion-MNIST
│   ├── train-images-idx3-ubyte/
│   ├── train-labels-idx1-ubyte/
│   ├── t10k-images-idx3-ubyte/
│   └── t10k-labels-idx1-ubyte/
└── README.md
```

## Compilación y Pruebas:

### Test Vision Transformer Completo:
```bash
./compile.sh
```

### Demo de Entrenamiento:
```bash
./train_demo.sh
```

### Tests Individuales:

#### Test Básico Transformer Layer:
```bash
g++ -std=c++17 -I. tests/01_test_transformer_layer.cpp src/matrix/matrix.cpp src/matrix/matrix_ops.cpp src/matrix/activation_functions.h.cpp src/transformer/multi_head_attention.cpp src/transformer/mlp.cpp src/transformer/layer_norm.cpp src/transformer/transformer_block.cpp -o test_transformer && ./test_transformer
```

#### Test Fashion-MNIST Data Loading:
```bash
g++ -std=c++17 -I. tests/03_test_fashion_vit.cpp src/matrix/matrix.cpp src/utils/file_io.cpp -o fashion_test_vit && ./fashion_test_vit
```

## Resultados de Pruebas:

### ✅ Test Vision Transformer Completo:
```
=== COMPLETE VISION TRANSFORMER TEST ===
Creating Vision Transformer...
- Image size: 28x28
- Patch size: 7x7
- Embed dim: 256
- Num heads: 8
- Num layers: 6
- Num classes: 10

Test images shape: 10000 x 784
Logits shape: 5 x 10
Predictions shape: 5 x 1
✅ Complete Vision Transformer test completed!
```

### ✅ Demo de Entrenamiento:
```
=== VISION TRANSFORMER TRAINING DEMO ===
Train images: 60000 x 784
Test images: 10000 x 784

=== TRAINING SIMULATION ===
Batch 1/5 - Loss: 5.62422, Accuracy: 18.75%
Batch 2/5 - Loss: 5.83847, Accuracy: 12.5%
...

=== TEST EVALUATION ===
Test Loss: 6.49407
Test Accuracy: 9%
✅ Vision Transformer training demo completed!
```

### ✅ Test Transformer Layer:
```
=== TRANSFORMER LAYER TEST ===
Creando Transformer Block...
- Embed dim: 256, Num heads: 8, MLP hidden: 1024
Input shape: 50 x 256 → Output shape: 50 x 256
✅ Transformer Layer funcionando correctamente!
```

## Configuración Típica:
- **Embed dimension**: 256
- **Number of heads**: 8
- **MLP hidden dimension**: 1024 (4x expansion)
- **Sequence length**: Variable
- **Fashion-MNIST**: 28x28 images (784 pixels)

## Uso:

### Vision Transformer Completo:
```cpp
#include "include/transformer/vision_transformer.h"
#include "include/transformer/loss_functions.h"
#include "include/utils/file_io.h"

// Crear Vision Transformer
VisionTransformer vit(28, 7, 256, 8, 1024, 6, 10);
//                   img_size, patch_size, embed_dim, num_heads, mlp_dim, num_layers, num_classes

// Cargar datos Fashion-MNIST
Matrix images = FileIO::load_mnist_images("data/train-images-idx3-ubyte/train-images-idx3-ubyte");
std::vector<int> labels = FileIO::load_mnist_labels("data/train-labels-idx1-ubyte/train-labels-idx1-ubyte");

// Forward pass
Matrix logits = vit.forward(images);
Matrix predictions = vit.get_predictions(logits);

// Evaluar
double loss = LossFunctions::cross_entropy_loss(logits, labels);
double accuracy = LossFunctions::accuracy(predictions, labels);
```

### Transformer Block Individual:
```cpp
#include "include/transformer/transformer_block.h"

// Crear transformer block
TransformerBlock transformer(256, 8, 1024);
Matrix output = transformer.forward(input);
```

## Características:

### Vision Transformer:
- ✅ **Patch Embedding** - Conversión de imágenes 28x28 a patches 7x7
- ✅ **CLS Token** - Token de clasificación prepended a la secuencia
- ✅ **Positional Encoding** - Embeddings posicionales aprendibles
- ✅ **Multi-layer Transformer** - Múltiples bloques transformer
- ✅ **Classification Head** - Capa final para 10 clases Fashion-MNIST
- ✅ **Loss Functions** - Cross-entropy loss y accuracy
- ✅ **Training Demo** - Simulación de entrenamiento con batches

### Transformer Components:
- ✅ Pre-norm architecture (LayerNorm antes de attention/MLP)
- ✅ Conexiones residuales
- ✅ Inicialización Xavier para pesos
- ✅ Activación GELU en MLP
- ✅ Softmax en attention
- ✅ Escalado por sqrt(head_dim) en attention
- ✅ Carga de datos MNIST/Fashion-MNIST binarios
- ✅ Tests funcionales verificados

## Fashion-MNIST Labels:
0: T-shirt/top, 1: Trouser, 2: Pullover, 3: Dress, 4: Coat,
5: Sandal, 6: Shirt, 7: Sneaker, 8: Bag, 9: Ankle boot