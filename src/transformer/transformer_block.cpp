#include "../../include/transformer/transformer_block.h"
#include "../../include/matrix/matrix_ops.h"

TransformerBlock::TransformerBlock(size_t embed_dim, size_t num_heads, size_t mlp_hidden_dim)
    : attention(embed_dim, num_heads), mlp(embed_dim, mlp_hidden_dim), norm1(embed_dim), norm2(embed_dim) {
}

Matrix TransformerBlock::forward(const Matrix& input) {
    // First residual block: LayerNorm -> Attention -> Add
    Matrix normed1 = norm1.forward(input);
    Matrix attn_out = attention.forward(normed1);
    
    // Residual connection
    Matrix residual1 = MatrixOps::add(input, attn_out);
    
    // Second residual block: LayerNorm -> MLP -> Add  
    Matrix normed2 = norm2.forward(residual1);
    Matrix mlp_out = mlp.forward(normed2);
    
    // Residual connection
    Matrix output = MatrixOps::add(residual1, mlp_out);
    
    return output;
}