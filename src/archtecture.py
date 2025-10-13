import torch
import torch.nn as nn
import torch.optim as optim
import tokenization
import math
from torch.nn import functional as F

class Scaled_dotProductAttention(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.scale = 1.0 / math.sqrt(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale  # (b, q_len, k_len)

        if attn_mask is not None:
            scores = scores + attn_mask

        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1)  # (b, 1, k_len)
            scores = scores.masked_fill(mask, float("-1e9"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, value)  # (b, q_len, d_model)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj   = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        # (batch, seq_len, d_model) -> (batch, n_heads, seq_len, d_head)
        b, s, _ = x.size()
        x = x.reshape(b, s, self.n_heads, self.d_head).transpose(1, 2)  # (b, h, s, d_head)
        return x

    def _combine_heads(self, x):
        # (batch, n_heads, seq_len, d_head) -> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous()  # (b, s, h, d_head)
        b, s, _, _ = x.size()
        return x.view(b, s, self.d_model)

    def forward(self, query, key, value, attn_mask=None, return_attn=False):
        # (batch, seq_len, d_model)
        Q = self._split_heads(self.query_proj(query))  # (b, h, q_len, d_head)
        K = self._split_heads(self.key_proj(key))      # (b, h, k_len, d_head)
        V = self._split_heads(self.value_proj(value))  # (b, h, k_len, d_head)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)  # (b, h, q_len, k_len)
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(~attn_mask, float('-1e9'))
            else:
                scores = scores + attn_mask

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, V)  # (b, h, q_len, d_head)
        context = self._combine_heads(context)  # (b, q_len, d_model)
        out = self.out_proj(context)

        if return_attn:
            return out, attn
        return out

class FeedForward(nn.Module):
    def __init__(self, d_model: int, dim_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dim_ff: int, dropout: float = 0.1, eps: float = 1e-5):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln1 = nn.LayerNorm(d_model, eps=eps)
        self.ff = FeedForward(d_model, dim_ff, dropout)
        self.ln2 = nn.LayerNorm(d_model, eps=eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        x2 = self.ln1(x)
        sa = self.mha(x2, x2, x2, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = x + self.dropout(sa)

        x2 = self.ln2(x)
        ff = self.ff(x2)
        x = x + self.dropout(ff) # (b, seq_len, d_model)
        return x
    
def causal_mask(seq_len: int, device: torch.device):
    mask = torch.triu(torch.full((seq_len, seq_len), float("-1e9")), diagonal=1).to(device)
    return mask


class GenerativeModel(nn.Module):
    """
    Expected cfg fields: vocab_size, d_model, n_heads, dim_ff, max_len, padding_idx, num_layers (optional), dropout (optional)
    """
    def __init__(self, cfg):
        super().__init__()
        n_layers = getattr(cfg, "num_layers", 10)
        dropout = getattr(cfg, "dropout", 0.1)

        padding_idx = getattr(cfg, "padding_idx", None)
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=padding_idx)
        self.pos_embed = tokenization.PositionalEncoding(d_model=cfg.d_model, max_len=getattr(cfg, "max_len", 512))
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model=cfg.d_model, n_heads=cfg.n_heads, dim_ff=cfg.dim_ff, dropout=dropout)
            for _ in range(n_layers)
        ])
        
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if getattr(cfg, "tie_weights", False):
            self.lm_head.weight = self.token_embed.weight

        self.d_model = cfg.d_model
        self.n_layers = n_layers
        self.padding_idx = padding_idx
    
    def forward(self, input_ids: torch.Tensor):
        """
        input_ids: (batch, seq_len) long
        returns: logits (batch, seq_len, vocab_size)
        """
        device = input_ids.device
        b, seq_len = input_ids.size()
        
        x = self.token_embed(input_ids) * math.sqrt(self.d_model)  # (b, seq_len, d_model)
        x = self.pos_embed(x)
        x = self.dropout(x)

        attn_mask = causal_mask(seq_len, device)
        key_padding_mask = None
        if self.padding_idx is not None:
            key_padding_mask = (input_ids == self.padding_idx)  # (b, seq_len) bool

        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

        logits = self.lm_head(x)  # (b, seq_len, vocab_size)
        return logits