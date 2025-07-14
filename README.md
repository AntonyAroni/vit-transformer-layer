# Transformer Layer Implementation

## Descripción
Implementación completa de una capa Transformer en C++ que incluye:

### Componentes Principales:
1. **Multi-Head Self-Attention** - Mecanismo de atención con múltiples cabezas
2. **MLP (Feed-Forward Network)** - Red neuronal feed-forward con activación GELU
3. **Layer Normalization** - Normalización de capas
4. **Transformer Block** - Capa completa con conexiones residuales
5. **File I/O Utilities** - Carga de datos MNIST/Fashion-MNIST

### Arquitectura:
```
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
│   │   └── transformer_block.h
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
│   │   └── transformer_block.cpp
│   └── utils/                     # Implementaciones utilidades
│       └── file_io.cpp
├── tests/
│   ├── 01_test_transformer_layer.cpp
│   ├── 02_fashion_mnist_example.cpp
│   └── 03_test_fashion_vit.cpp
├── data/                          # Datos Fashion-MNIST
│   ├── train-images-idx3-ubyte/
│   ├── train-labels-idx1-ubyte/
│   ├── t10k-images-idx3-ubyte/
│   └── t10k-labels-idx1-ubyte/
└── README.md
```

## Compilación y Pruebas:

### Test Básico Transformer:
```bash
g++ -std=c++17 -I. tests/01_test_transformer_layer.cpp src/matrix/matrix.cpp src/matrix/matrix_ops.cpp src/matrix/activation_functions.h.cpp src/transformer/multi_head_attention.cpp src/transformer/mlp.cpp src/transformer/layer_norm.cpp src/transformer/transformer_block.cpp src/utils/file_io.cpp -o test_transformer && ./test_transformer
```

### Test Fashion-MNIST:
```bash
g++ -std=c++17 -I. tests/03_test_fashion_vit.cpp src/matrix/matrix.cpp src/utils/file_io.cpp -o fashion_test_vit && ./fashion_test_vit
```

## Resultados de Pruebas:

### ✅ Test Transformer Layer:
```
=== TRANSFORMER LAYER TEST ===
Creando Transformer Block...
- Embed dim: 256
- Num heads: 8
- MLP hidden: 1024

Input shape: 50 x 256
Ejecutando forward pass...
Output shape: 50 x 256
✅ Transformer Layer funcionando correctamente!
```

### ✅ Test Fashion-MNIST:
```
Fashion-MNIST test set: 10000 images
Image dimensions: 784 pixels
First 10 samples: Ankle boot, Pullover, Trouser, etc.
✅ Fashion-MNIST data loaded successfully!
```

## Configuración Típica:
- **Embed dimension**: 256
- **Number of heads**: 8
- **MLP hidden dimension**: 1024 (4x expansion)
- **Sequence length**: Variable
- **Fashion-MNIST**: 28x28 images (784 pixels)

## Uso:
```cpp
#include "include/transformer/transformer_block.h"
#include "include/utils/file_io.h"

// Crear transformer block
TransformerBlock transformer(256, 8, 1024);

// Cargar datos Fashion-MNIST
Matrix images = FileIO::load_mnist_images("data/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte");
std::vector<int> labels = FileIO::load_mnist_labels("data/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte");

// Forward pass
Matrix output = transformer.forward(input);
```

## Características:
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