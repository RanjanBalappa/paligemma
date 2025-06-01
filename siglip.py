import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


#config class
class SiglipVisionConfig:
    def __init__(self,
                 image_size: int = 224,
                 patch_size: int = 16,
                 num_channels: int = 3,
                 hidden_size: int = 768,
                 num_hidden_layers: int = 12,
                 num_attention_heads: int = 12,
                 layer_norm_eps: float = 1e-6,
                 attention_dropout: float = 0.0,
                 num_image_tokens: int = None,
                 **kwargs):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens


#vision embedding layer which contains conv2d for patch embedding and position embedding
class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        image_size = config.image_size
        patch_size = config.patch_size

        self.patch_embeddings = nn.Conv2d(
            input_channels=config.num_channels,
            output_channels=config.hidden_state,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            padding=0,
        )
        self.num_patches = (image_size // patch_size) ** 2
        self.num_positions = self.num_patches 
        self.position_embeddings = nn.Embedding(self.num_positions, config.hidden_size)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1))
        )

    def forward(self, pixel_values: Tensor):
        patch_embeddings = self.patch_embeddings(pixel_values) #[batch_size, hidden_size, num_patches_H, num_patches_W]
        embeddings = patch_embeddings.flatten(2) #[batch_size, hidden_size, num_patches]
        embeddings = embeddings.transpose(1, 2) #[batch_size, num_patches , hidden_size]
        embeddings = embeddings + self.position_embeddings(self.position_ids) #[batch_size, num_patches , hidden_size]
        return embeddings
    

#mlp layer
class SiglipMLP(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: Tensor):
        hidden_states = self.fc1(hidden_states)
        hidden_states = F.gelu(hidden_states, appoximate='tanh')
        hidden_states = self.fc2(hidden_states)
        return hidden_states
    

#self attention layer
class SelfAttention(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Liear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
    
    def forward(self, hidden_states: Tensor):
        batch_size, seq_len, _ = hidden_states.shape
       
        query_states = self.q_proj(hidden_states) #[batch_size, num_patches, embed_dim]
        key_states = self.k_proj(hidden_states) #[batch_size, num_patches, embed_dim]
        value_states = self.v_proj(hidden_states) #[batch_size, num_patches, embed_dim]

        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) #[batch_size, num_heads, num_patches, head_dim]
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) #[batch_size, num_heads, num_patches, head_dim]
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) #[batch_size, num_heads, num_patches, head_dim]

        attn_weights = query_states @ key_states.transpose(-2, -1) * self.scale #[batch_size, num_heads, num_patches, num_patches]
        attn_weights = F.softmax(attn_weights, dim=-1) #[batch_size, num_heads, num_patches, num_patches]
        attn_weights = self.dropout(attn_weights)

        attn_output = attn_weights @ value_states #[batch_size, num_heads, num_patches, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous() #[batch_size, num_patches, num_heads, head_dim]
        attn_output = attn_output.view(batch_size, seq_len, self.embed_dim) #[batch_size, num_patches, embed_dim]
        attn_output = self.out_proj(attn_output) #[batch_size, num_patches, embed_dim]
        return attn_output, attn_weights

    def forward(self, hidden_states: Tensor):
        batch_size, seq_len, _ = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, seq_len, self.num_heads, self.head_dim)
        hidden_states = hidden_states.transpose(1, 2)




#Encoder Layer which has layer norm, self attention and mlp
class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.self_attn = SelfAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: Tensor):
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states) #[batch_size, num_patches , hidden_size]
        hidden_states, _ = self.self_attn(hidden_states) #[batch_size, num_patches, hidden_size]
        hidden_states = hidden_states + residual #[batch_size, num_patches, hidden_size]
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states) #[batch_size, num_patches, hidden_size]
        hidden_states = self.mlp(hidden_states) #[batch_size, num_patches, hidden_size]
        hidden_states = hidden_states + residual #[batch_size, num_patches, hidden_size]
        return hidden_states

class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config 
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(layer) for layer in range(config.num_hidden_layers)]
        )
    def forward(self, hidden_states: Tensor):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states
       
class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, pixel_values: Tensor):
        hidden_states = self.embeddings(pixel_values)
        last_hidden_state = self.encoder(hidden_states)
        last_hidden_state = self.post_layernorm(last_hidden_state)
        return last_hidden_state


class SiglipVisionModel(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values: Tensor):
        return self.vision_model(pixel_values=pixel_values)
        