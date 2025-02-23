import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, d_embed, d_head=64, num_heads=1, in_proj_bias=False, out_proj_bias=False):
        super(SelfAttention, self).__init__()
        self.d_embed = d_embed
        self.d_head = d_head
        self.num_heads = num_heads
        self.proj_dim = d_head * num_heads
        
        # Define projection layers for queries, keys, and values
        self.q_proj = nn.Linear(d_embed, self.proj_dim, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_embed, self.proj_dim, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_embed, self.proj_dim, bias=in_proj_bias)
        self.out_proj = nn.Linear(self.proj_dim, self.proj_dim, bias=out_proj_bias)
        
        self._init_weights()
    
    def _init_weights(self):
        # Initialize weights using Xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.1)
    
    @staticmethod
    def masked_softmax(attn_logits, attention_mask):
        # Apply softmax with masking support
        softmax = nn.Softmax(dim=-1)
        if attention_mask is None:
            return softmax(attn_logits)
        else:
            attn_logits_masked = attn_logits.masked_fill(attention_mask == 0, -1e12)
            return softmax(attn_logits_masked)
    
    def forward(self, x, attention_mask=None):
        # Get batch size and sequence length
        batch_size, seq_length, _ = x.shape
        
        # Compute queries, keys, and values
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape and permute for multi-head attention
        q, k, v = [
            tensor.view(batch_size, seq_length, self.num_heads, self.d_head).permute(0, 2, 1, 3)
            for tensor in (q, k, v)
        ]
        
        # Compute attention scores
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        
        # Apply masked softmax to obtain attention weights
        attn_weights = self.masked_softmax(attn_logits, attention_mask)
        
        # Compute attention output
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(batch_size, seq_length, self.proj_dim)
        
        return self.out_proj(attn_output), attn_weights

class CrossAttention(nn.Module):
    def __init__(self, d_embed, d_cross, d_head=64, num_heads=1, in_proj_bias=False, out_proj_bias=True):
        super(CrossAttention, self).__init__()
        self.d_embed = d_embed
        self.d_cross = d_cross
        self.d_head = d_head
        self.num_heads = num_heads
        self.proj_dim = d_embed
        
        # Define projection layers for queries, keys, and values from different sources
        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
    
    def forward(self, x, y):
        # Get batch size and sequence length
        batch_size, seq_length, _ = x.shape
        
        # Compute queries from x, keys and values from y
        q = self.q_proj(x).view(batch_size, seq_length, self.num_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(y).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(y).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        
        # Compute attention scores
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_logits, dim=-1)
        
        # Compute attention output
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_length, self.proj_dim)
        
        return self.out_proj(attn_output), attn_weights
