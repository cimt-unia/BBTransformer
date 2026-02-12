# model.py 

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Any, List


# ======================
# TRANSFORMER COMPONENTS
# ======================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
    
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self._cache = {}  # Cache for RoPE embeddings

    def get_cos_sin(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get or compute rotary embeddings with caching"""
        cache_key = (seq_len, str(device))
        if cache_key not in self._cache:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat([freqs, freqs], dim=-1)
            self._cache[cache_key] = (emb.cos(), emb.sin())
        return self._cache[cache_key]

    def forward(self, seq_len, device):
        return self.get_cos_sin(seq_len, device)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ======================
# PATCH EMBEDDING
# ======================
class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, in_dim, embed_dim, dropout=0.1):
        """Patch embedding with parameterized dropout and activation"""
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size * in_dim, embed_dim, bias=False)
        self.norm = RMSNorm(embed_dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape
        
        # Handle sequences not divisible by patch_size
        if T % self.patch_size != 0:
            trim = T % self.patch_size
            x = x[:, :-trim]
            T = x.shape[1]  # Update T after trimming
        
        # Reshape into patches and project (using reshape instead of view for safety)
        x = x.reshape(B, T // self.patch_size, self.patch_size * D)
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class GQAWithRoPE(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads=None, dropout=0.1, return_attn_weights=False):
        """Grouped-Query Attention with Rotary Positional Embeddings and optional attention weights"""
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or max(1, n_heads // 4)
        self.return_attn_weights = return_attn_weights
        
        # Ensure n_kv_heads is valid
        if self.n_kv_heads > n_heads:
            self.n_kv_heads = n_heads
        assert n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        
        self.head_dim = d_model // n_heads
        self.dropout = dropout

        # Projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Rotary embeddings with caching
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x):
        B, L, D = x.shape
        
        # Project and reshape
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE with caching
        cos, sin = self.rope(L, x.device)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Expand k/v to match q heads (GQA)
        if self.n_heads != self.n_kv_heads:
            k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
            v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)

        # Compute attention with optional weights
        if self.return_attn_weights and hasattr(F, 'scaled_dot_product_attention'):
            # PyTorch 2.0+ with attention weights
            attn_output, attn_weights = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
                return_attn_weights=True
            )
            self.last_attn_weights = attn_weights.detach()
        else:
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False
            )
            self.last_attn_weights = None

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(attn_output)



# ======================
#  MODEL ARCHITECTURE
# ======================
class BBTransformer(nn.Module):
    def __init__(self, 
                 feature_dim, 
                 num_classes=1, 
                 embed_dim=512,          
                 num_heads=8,
                 num_layers=6, 
                 dropout_input=0.27,
                 dropout_attn=0.15,
                 dropout_ffn=0.28,
                 dropout_classifier=0.03,
                 dropout_temporal=0.17,
                 embed_dim_age=32,       
                 embed_dim_ext=16,
                 patch_size=3,          
                 patch_embed_ratio=0.5,
                 temp_attn_hidden=128,  
                 n_kv_heads=4,          
                 return_attn_weights=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.patch_embed_dim = int(embed_dim * patch_embed_ratio)
        self.return_attn_weights = return_attn_weights

        # --- Input projection ---
        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, embed_dim, bias=False),
            RMSNorm(embed_dim),
            nn.Dropout(dropout_input)
        )

        # --- Patch embedding (short-scale) ---
        self.patch_embed = PatchEmbedding(
            patch_size=patch_size,
            in_dim=embed_dim,
            embed_dim=self.patch_embed_dim,
            dropout=dropout_input  # or use separate patch_dropout if needed
        )

        # --- Temporal attention for global pooling ---
        self.temporal_attn = nn.Sequential(
            nn.Linear(embed_dim, temp_attn_hidden, bias=True),
            nn.Tanh(),
            nn.Dropout(dropout_temporal),
            nn.Linear(temp_attn_hidden, 1, bias=True)
        )
        
        self.last_attention_maps = {}

        # --- GQA configuration ---
        if n_kv_heads is None:
            n_kv_heads = num_heads // 4 if num_heads >= 8 else max(1, num_heads // 2)
        self.n_kv_heads = n_kv_heads

        # --- Transformer layers ---
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = nn.ModuleDict({
                'norm1': RMSNorm(embed_dim),
                'attn': GQAWithRoPE(
                    d_model=embed_dim,
                    n_heads=num_heads,
                    n_kv_heads=n_kv_heads,
                    dropout=dropout_attn,
                    return_attn_weights=return_attn_weights
                ),
                'norm_post_attn': RMSNorm(embed_dim),
                'norm2': RMSNorm(embed_dim),
                'ffn': SwiGLU(embed_dim, embed_dim * 4),
                'norm_post_ffn': RMSNorm(embed_dim),
                'dropout': nn.Dropout(dropout_ffn)
            })
            self.layers.append(layer)

        # --- Cross-attention ---
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout_attn, batch_first=True
        )
        self.cross_norm = RMSNorm(embed_dim)

        # --- Confounder embeddings ---
        self.age_proj = nn.Linear(1, embed_dim_age)
        self.ext_embed = nn.Embedding(2, embed_dim_ext)

        # --- Final classifier ---
        total_input = embed_dim + self.patch_embed_dim + embed_dim_age + embed_dim_ext
        self.classifier = nn.Sequential(
            nn.Linear(total_input, embed_dim, bias=False),
            RMSNorm(embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout_classifier),
            nn.Linear(embed_dim, embed_dim // 2, bias=False),
            RMSNorm(embed_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_classifier * 0.5),
            nn.Linear(embed_dim // 2, num_classes, bias=True)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, x, age, ext):
        B, T, D = x.shape
        
        # Input projection
        x = self.input_proj(x)

        # Short-scale
        x_short = self.patch_embed(x)
        global_short = x_short.mean(dim=1)

        # Main transformer
        if self.return_attn_weights:
            self.last_attention_maps = {'main': [], 'cross': None}
        
        for i, layer in enumerate(self.layers):
            # Attention
            residual = x
            x = layer['norm1'](x)
            x = layer['attn'](x)
            x = layer['dropout'](x)
            x = residual + x
            x = layer['norm_post_attn'](x)

            # FFN
            residual = x
            x = layer['norm2'](x)
            x = layer['ffn'](x)
            x = layer['dropout'](x)
            x = residual + x
            x = layer['norm_post_ffn'](x)
            
            if self.return_attn_weights and hasattr(layer['attn'], 'last_attn_weights'):
                if layer['attn'].last_attn_weights is not None:
                    self.last_attention_maps['main'].append(
                        layer['attn'].last_attn_weights.cpu().detach()
                    )

        # Multi-scale fusion
        x_short_up = F.interpolate(
            x_short.transpose(1, 2), size=T, mode='linear', align_corners=False
        ).transpose(1, 2)

        pad_dim = self.embed_dim - x_short_up.size(-1)
        if pad_dim > 0:
            padding = torch.zeros(B, T, pad_dim, device=x.device, dtype=x.dtype)
            x_short_padded = torch.cat([x_short_up, padding], dim=-1)
        else:
            x_short_padded = x_short_up[:, :, :self.embed_dim]

        if self.return_attn_weights:
            cross_attended, cross_attn_weights = self.cross_attn(x, x_short_padded, x_short_padded)
            self.last_attention_maps['cross'] = cross_attn_weights.cpu().detach()
        else:
            cross_attended, _ = self.cross_attn(x, x_short_padded, x_short_padded)
        
        x = self.cross_norm(x + cross_attended)

        # Temporal pooling
        attn_weights = F.softmax(self.temporal_attn(x), dim=1)
        global_main = torch.sum(x * attn_weights, dim=1)

        # Confounders
        age_emb = self.age_proj(age.unsqueeze(1))
        ext_emb = self.ext_embed(ext)

        # Final fusion
        combined = torch.cat([global_main, global_short, age_emb, ext_emb], dim=1)
        logits = self.classifier(combined).squeeze(-1)
        return logits

    def get_attention_maps(self) -> Optional[Dict[str, Any]]:
        return self.last_attention_maps if self.return_attn_weights else None

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)




# ======================
# CONFIGURATION
# ======================
def create_bbtransformer(config: dict) -> BBTransformer:
    """Factory function to create a BBTransformer with full configuration support"""
    default_config = {
        # Core
        'feature_dim': 414,
        'num_classes': 1,
        'embed_dim': 256,
        'num_heads': 8,
        'num_layers': 6,
        # Dropouts (decoupled)
        'dropout_input': 0.1,
        'dropout_attn': 0.1,
        'dropout_ffn': 0.1,
        'dropout_classifier': 0.1,
        'dropout_temporal': 0.05,
        # Confounder embeddings
        'embed_dim_age': 16,
        'embed_dim_ext': 16,
        # Multi-scale
        'patch_size': 2,
        'patch_embed_ratio': 0.5,
        # Temporal attention
        'temp_attn_hidden': 64,
        # GQA
        'n_kv_heads': None,  # Auto-computed if None
        # Debug/interpretability
        'return_attn_weights': False
    }
    
    # Override defaults with user config
    default_config.update(config)
    
    # Pass all relevant args to BBTransformer
    return BBTransformer(
        feature_dim=default_config['feature_dim'],
        num_classes=default_config['num_classes'],
        embed_dim=default_config['embed_dim'],
        num_heads=default_config['num_heads'],
        num_layers=default_config['num_layers'],
        dropout_input=default_config['dropout_input'],
        dropout_attn=default_config['dropout_attn'],
        dropout_ffn=default_config['dropout_ffn'],
        dropout_classifier=default_config['dropout_classifier'],
        dropout_temporal=default_config['dropout_temporal'],
        embed_dim_age=default_config['embed_dim_age'],
        embed_dim_ext=default_config['embed_dim_ext'],
        patch_size=default_config['patch_size'],
        patch_embed_ratio=default_config['patch_embed_ratio'],
        temp_attn_hidden=default_config['temp_attn_hidden'],
        n_kv_heads=default_config['n_kv_heads'],
        return_attn_weights=default_config['return_attn_weights']
    )



def load_pretrained_bbtransformer(weights_path: str, config: dict) -> BBTransformer:
    """Load a pretrained BBTransformer model"""
    model = create_bbtransformer(config)
    
    # Load weights
    state_dict = torch.load(weights_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    
    # Filter out keys that don't match
    model_state = model.state_dict()
    pretrained_state = {k: v for k, v in state_dict.items() if k in model_state and v.shape == model_state[k].shape}
    
    # Update the model's state dict
    model_state.update(pretrained_state)
    model.load_state_dict(model_state)
    
    return model


