import torch
import torch.nn as nn
import torch.nn.functional as F

class NumericalEmbedding(nn.Module):
    def __init__(self, num_features, embed_dim):
        super().__init__()
        self.num_features = num_features
        self.embed_dim = embed_dim
        
        # 1. Use Conv1d with groups=num_features to achieve per-feature linear projection
        # Equivalent to having separate Linear layers for each feature, but more efficient
        # Like input with shape [batch, num_features, 1], it got groups * [1, embed_dim] of non-shared weight matrix
        self.numerical_embedding = nn.Conv1d(
            in_channels=num_features,
            out_channels=num_features * embed_dim,
            kernel_size=1,
            groups=num_features,
        )
        
        # 2. Layer Normalization
        # The key is to do Norm over the embedding dimension
        # Ensuring that the embedding vectors for each feature have controlled magnitude, independent of batch distribution
        self.layernorm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        # x: [batch_size, num_features]
        
        # --- Step 1: Signed Log ---
        # Reference: Mastering Diverse Domains through World Models (DreamerV3)
        # This to replace Standardization (Mean/Std)
        # Ensuring the model can handle wide-ranging numerical values without instability
        # Even the input is 100 million, after log it's about 20, which neural networks can handle
        x = torch.sign(x) * torch.log1p(torch.abs(x))
                
        # --- Step 2: Numerical Embedding ---
        x = x.unsqueeze(-1) # [batch, num_features] -> [batch, num_features, 1]
        x = self.numerical_embedding(x) # -> [batch, num_features * embed_dim, 1]
        x = x.view(-1, self.num_features, self.embed_dim) # -> [batch, num_features, embed_dim]
        
        # --- Step 3: Layer Normalization ---
        # Treat is as batch * num_features of embed_dim vectors to normalize individually
        # This ensures the normalization is per-feature embedding, not across features
        # Even batch size is 1, it still works correctly
        x = self.layernorm(x)
        
        return x


class PromptEmbeddings(nn.Module):
    def __init__(self, num_features, embed_dim):
        super().__init__()
        self.prompt_embeddings = nn.Embedding(num_features, embed_dim).weight
    
    def forward(self, x):
        # Expand based on batch size
        return self.prompt_embeddings.expand(x.shape[0], -1, -1)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_dim, head_dim, attention_bias=False, attention_dropout=0.0):
        super().__init__()
        
        # Check that hidden_dim is divisible by head_dim
        if hidden_dim % head_dim != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by head_dim ({head_dim})"
            )
        
        # Hyperparameters used in forward pass
        self.num_attention_heads = hidden_dim // head_dim
        self.head_dim = head_dim
        self.scaling = head_dim**-0.5
        
        # Learnable projections for Q, K, V
        self.q_proj = nn.Linear(
            hidden_dim, self.num_attention_heads * self.head_dim, bias=attention_bias,
        )
        self.k_proj = nn.Linear(
            hidden_dim, self.num_attention_heads * self.head_dim, bias=attention_bias,
        )
        self.v_proj = nn.Linear(
            hidden_dim, self.num_attention_heads * self.head_dim, bias=attention_bias,
        )
        
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, hidden_states):
        batch_size, num_features, _ = hidden_states.shape
        
        # --- Step 1: Linear Projections ---
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # --- Step 2: Reshape & Transpose (Multi-Head) ---
        # Decompose hidden_dim into num_heads * head_dim
        # (B, N, H*D) -> (B, N, H, D) -> (B, H, N, D)
        query_states = query_states.view(batch_size, num_features, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, num_features, self.num_attention_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, num_features, self.num_attention_heads, self.head_dim).transpose(1, 2)
        
        # --- Step 3: Scaled Dot-Product Attention ---
        # Q * K^T / sqrt(d)
        # Q: (B, H, N, D) K^T: (B, H, D, N)
        # attn_weights: (B, H, N, N) describing attention scores between all feature pairs
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        # Softmax normalization
        attn_weights = F.softmax(attn_weights, dim=-1)
        # Attention dropout
        attn_weights = self.dropout(attn_weights)
        
        # --- Step 4: Weighted Sum (Aggregate Values) ---
        # attn_weights * V: (B, H, N, N) * (B, H, N, D) -> (B, H, N, D)
        attn_output = torch.matmul(attn_weights, value_states)
        
        # --- Step 5: Restore Shape (Concat Multi-Head) ---
        # (B, H, N, D) -> (B, N, H, D) -> (B, N, H*D)
        # Notice: after transpose, memory is not contiguous, must call .contiguous() before view
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, num_features, -1)
        
        return attn_output, attn_weights


class NumericalProjector(nn.Module):
    def __init__(self, hidden_dim, embed_dim, projector_bias):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, embed_dim, bias=projector_bias)
    
    def forward(self, x):
        return self.linear(x)


class NumericalPromptEncoder(nn.Module):
    def __init__(
        self, 
        use_numerical_embedding,
        use_multi_head_self_attn,
        use_numerical_projector,
        num_features, 
        embed_dim, 
        head_dim=128, 
        attention_bias=False, 
        projector_bias=False, 
        attention_dropout=0.0, 
    ):
        super().__init__()
        # whether to use numerical embeddings
        if use_numerical_embedding:
            self.numerical_embedding = NumericalEmbedding(
                num_features=num_features, 
                embed_dim=embed_dim,
            )
        else:
            self.numerical_embedding = PromptEmbeddings(
                num_features=num_features,
                embed_dim=embed_dim,
            )
        # whether to use multi-head self-attention
        if use_multi_head_self_attn:
            self.multi_head_self_attention = MultiHeadSelfAttention(
                hidden_dim=embed_dim,
                head_dim=head_dim,
                attention_bias=attention_bias,
                attention_dropout=attention_dropout,
            )
        else:
            self.multi_head_self_attention = lambda x: (x, None)
        # whether to use numerical projector
        if use_numerical_projector:
            self.numerical_projector = NumericalProjector(
                hidden_dim=embed_dim, 
                embed_dim=embed_dim, 
                projector_bias=projector_bias,
            )
        else:
            self.numerical_projector = lambda x: x
    
    def forward(self, x):
        # x: [batch_size, num_features]
        
        # 1. Numerical Embedding
        x = self.numerical_embedding(x) # [batch_size, num_features, embed_dim]
        
        # 2. Multi-Head Self-Attention
        x, attn_weights = self.multi_head_self_attention(x) # [batch_size, num_features, embed_dim]
        
        # 3. Numerical Projector
        x = self.numerical_projector(x) # [batch_size, num_features, embed_dim]
        
        return x


if __name__ == "__main__":
    # Numerical Embedding Test
    model = NumericalEmbedding(num_features=5, embed_dim=16)

    # One sample with numerical features in different magnitudes
    input_one = torch.tensor([[100000.0, 0.05, -500.0, 3.0, 0.0]])

    output = model(input_one)

    print("Output shape:", output.shape) # [1, 5, 16]
    print("Contain NaN?", torch.isnan(output).any().item())
    print("Mean (approx 0):", output.mean().item()) # LayerNorm ensure the mean is around 0
    
    # Multi-Head Self-Attention Test
    batch_size = 2
    num_features = 23
    hidden_dim = 4096
    head_dim = 128
    input_tensor = torch.randn(batch_size, num_features, hidden_dim)
    
    # Initialize Multi-Head Self-Attention
    attention = MultiHeadSelfAttention(hidden_dim, head_dim)
    
    # Forward pass
    output = attention(input_tensor)
    
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output.shape)
    assert input_tensor.shape == output.shape
