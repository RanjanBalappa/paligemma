from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from siglip import SiglipVisionConfig, SiglipVisionModel


####################KV Cache######################################
class KVCache:
    def __init__(self):
        self.key_cache: List[Tensor] = []
        self.value_cache: List[Tensor] = []

    def num_items(self):
        if len(self.key_cache) == 0:
            return 0
        return self.key_cache[0].shape[-2] #kv chanche is if dim [batch_size, num_heads, seq_len, head_dim]
    
    def update(self, key_states: Tensor, value_states: Tensor, layer_idx: int):
        if layer_idx >= len(self.key_cache):
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]
    
    def get(self, layer_idx: int):
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

####################Config######################################
class GemmaConfig():
    def __init__(
        self,
        vocab_size,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim=256,
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id

class PaligemmaConfig:
    def __init__(
            self, 
            vision_config: SiglipVisionConfig,
            text_config: GemmaConfig,
            vocab_size: int = 257152,
            ignore_index=-100,
            image_token_index=256000,
            projection_dim=2048,
            hidden_size=2048,
            pad_token_id=None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size    
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.pad_token_id = pad_token_id
        self.vision_config = SiglipVisionConfig(**vision_config)
        self.text_config = text_config
        self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)
        self.vision_config.projection_dim = projection_dim
        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2


####################Projector for image features to convert to same dimenion required by text model######################################
class SiglipMultiModalProjector(nn.Module):
    def __init__(self, config: PaligemmaConfig):
        super().__init__()
        self.linear = nn.Linear(self.config.hidden_size, self.config.projection_dim, bias=True)

    def forward(self, image_features: Tensor):
        return self.linear(image_features)
    


####################RMSNorm######################################
class GemmaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def _norm(self, x: Tensor):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
    
    def forward(self, x: Tensor):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float()) 
        return output 
    

####################Rotary Positional Embedding####################
def rotate_half(x):
    # Build the [-x2, x1, -x4, x3, ...] tensor for the sin part of the positional encoding.
    x1 = x[..., : x.shape[-1] // 2] # Takes the first half of the last dimension
    x2 = x[..., x.shape[-1] // 2 :] # Takes the second half of the last dimension
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim) # Add the head dimension
    sin = sin.unsqueeze(unsqueeze_dim) # Add the head dimension
    # Apply the formula (34) of the Rotary Positional Encoding paper.
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class GemmaRotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_position_embeddings: int, base: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        inv_freq = 1.0 / (self.base ** (2 * torch.arange(0, self.head_dim, 2, dtupe=torch.int64).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: Tensor, position_ids: Optional[Tensor] = None, seq_len: Optional[int] = None):
        self.inv_freq = self.inv_freq.to(x.device)
        inv_freq_expanded = self.inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1) # [batch_size, head_Dim // 2, 1]
        position_ids = position_ids[:, None, :].float() # [batch_size, 1, seq_len]
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids.float()).transpose(1, 2) # [batch_size, seq_len, head_dim // 2]
            freqs = torch.cat([freqs, freqs], dim=-1) # [batch_size, seq_len, head_dim] #slight modification as per HF
            cos = freqs.cos()
            sin = freqs.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


####################Attention######################################
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class GemmaAttention(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_key_value_heads // self.head_dim
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.layer_idx = layer_idx
        self.is_causal = True

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=True)

        self.rotary_emb = GemmaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings, base=self.rope_theta)

    def forward(
            self,
            hidden_states: Tensor,
            attention_mask: Optional[Tensor] = None,
            position_ids: Optional[Tensor] = None,
            kv_cache: Optional[KVCache] = None,
    ):
        batch_size, q_len, _ = hidden_states.shape
        dtype = hidden_states.dtype
        device = hidden_states.device

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        v_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v_states = v_states.view(batch_size, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        #Rotary positional embedding
        cos, sin = self.rotary_emb(v_states, position_ids, seq_len=None)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if kv_cache is not None:
            key_states, value_states = kv_cache.update(key_states, value_states, self.layer_idx)

        # Repeat the key and values to match the number of heads of the query
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        assert attention_mask is not None
        attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights

      

####################MLP######################################
class GemmaMLP(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=True)
        
    def forward(self, x: Tensor):
        return self.down_proj(F.gelu(self.up_proj(x) * self.gate_proj(x)))
    
####################Decoder Layer######################################
class GemmaDecoderLayer(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.config = config

        self.self_attn = GemmaAttention(config, layer_idx=layer_idx)
        self.mlp = GemmaMLP(config)
        self.norm1 = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm2 = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
            self,
            hidden_states: Tensor,
            attention_mask: Optional[Tensor] = None,
            position_ids: Optional[Tensor] = None,
            kv_cache: Optional[KVCache] = None,
    ):
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states, _,  = self.self_attn(hidden_states, attention_mask, position_ids, kv_cache)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

####################Model######################################
class GemmaModel(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(self.vocab_size, self.config.hidden_size, padding_idx=self.config.pad_token_id)
        self.layers = nn.ModuleList([
            GemmaDecoderLayer(config, layer_idx=layer_idx) for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self):
        return self.embed_tokens
    
    def forward(
            self,
            inputs_embeds: Optional[Tensor] = None,
            attention_mask: Optional[Tensor] = None,
            position_ids: Optional[Tensor] = None,
            kv_cache: Optional[KVCache] = None,
    ):
        hidden_states = inputs_embeds
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states, attention_mask, position_ids, kv_cache)

        hidden_states = self.norm(hidden_states)
        return hidden_states

####################For Causal LM######################################
class GemmaForCausalLM(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.model = GemmaModel(config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=True)

    def tie_weights(self):
        self.language_model.tie_weights()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def forward(
            self,
            inputs_embeds: Optional[Tensor] = None,
            attention_mask: Optional[Tensor] = None,
            position_ids: Optional[Tensor] = None,
            kv_cache: Optional[KVCache] = None,
    ):
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        
        hidden_states = outputs
        logits = self.lm_head(hidden_states)
        result = {
            "logits": logits,
        }
        
        if kv_cache is not None:
            result["kv_cache"] = kv_cache

        return result




class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config: PaligemmaConfig):
        super().__init__()
        self.config = config
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = SiglipMultiModalProjector(config)
        self.vocab_size = config.vocab_size
        
        language_model = GemmaForCausalLM(config.text_config)
        self.language_model - language_model

        self.pad_id = self.config.pad_id if self.config.pad_id is not None else -1

    def tie_weights(self):
        self.language_model.tie_weights()

    def _merge_image_and_text(
            self,
            input_embeds: Tensor,
            image_features: Tensor,
            attention_mask: Tensor,
            input_ids: Tensor,
            kv_cache: Optional[KVCache] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        batch_size, seq_len, _ = input_embeds.shape
        _, _, embed_dim = image_features.shape
        dtype, device = input_embeds.dtype, input_embeds.device
        scaled_image_featuers = image_features / self.config.hidden_size ** 0.5

        #crate a final embeddings
        final_embeddings = torch.zeros(
            batch_size,
            seq_len,
            embed_dim,
            dtype=dtype,
            device=device,
        )

        image_mask = input_ids == self.config.image_token_index
        text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.config.pad_token_id)
        pad_mask = input_ids == self.config.pad_token_id

        #expand mask to be used with where
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        final_embeddings = torch.where(text_mask_expanded, input_embeds, final_embeddings)
        final_embeddings = torch.where(pad_mask_expanded, torch.zeros_like(final_embeddings), final_embeddings)
        final_embeddings = final_embeddings.masked_scatter(image_mask_expanded, scaled_image_featuers)
        
        #create attention mask for scenarios of pre fill and autoregression
        q_len = input_embeds.shape[1]
        if kv_cache is None or kv_cache.num_items() == 0:
            causal_mask = torch.full(
                (batch_size, q_len, q_len),
                fill_value=0,
                dtype=dtype,

            )
        else:
            assert q_len == 1
            kv_len = kv_cache.num_items() + q_len
            causal_mask = torch.full(
                (batch_size, q_len, kv_len),
                fill_value=0,
                dtype=dtype,
            )
        #make causal mak to have head dim
        causal_mask = causal_mask.unsqueeze(1)

        #calculate query position ids in case of prefill position ids of all tokens but during token generation query position is only last dim
        if kv_cache is not None and kv_cache.num_items() > 0:
            positional_ids = attention_mask.cumsum(dim=-1)[:, -1]
            if positional_ids.ndim == 1:
                positional_ids = positional_ids.unsqueeze(1)
        else:
            positional_ids = attention_mask.cumsum(dim=-1).masked_fill_(attention_mask == 0, 1)

        return final_embeddings, causal_mask, positional_ids

    def forward(
            self, 
            pixel_values: Tensor,
            input_ids: Tensor,
            attention_mask: Optional[Tensor] = None,
            kv_cache: Optional[KVCache] = None,
    ) -> Tuple:
        
        #1: Extract input embeddings
        input_embeds = self.language_model.get_input_embeddings()(input_ids)

        #2: Mergge image and text
        selected_image_feature = self.vision_tower(pixel_values=pixel_values)
        image_features = self.multi_modal_projector(selected_image_feature)
        input_embeds, attention_mask, position_ids = self._merge_image_and_text(input_embeds, image_features, attention_mask, input_ids, kv_cache)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
        )

        return outputs

   