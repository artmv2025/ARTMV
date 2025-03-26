import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torch.nn.functional as F

import torchvision.datasets as dset
import torchvision.transforms as T

import numpy as np

class SelfAttention(nn.Module):
    def __init__(self, input_dims, head_dims=128, num_heads=2, bias=False):
        super(SelfAttention, self).__init__()
        self.input_dims = input_dims
        self.head_dims = head_dims
        self.num_heads = num_heads
        self.proj_dims = head_dims * num_heads
        
        self.W_Q = nn.Linear(input_dims, self.proj_dims, bias=bias)
        self.W_K = nn.Linear(input_dims, self.proj_dims, bias=bias)
        self.W_V = nn.Linear(input_dims, self.proj_dims, bias=bias)
        self.W_O = nn.Linear(self.proj_dims, self.proj_dims, bias=bias)
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.1)

    def forward(self, x):
        b, n, d = x.shape
        
        q_ = self.W_Q(x)
        k_ = self.W_K(x)
        v_ = self.W_V(x)
        
        q, k, v = map(lambda z: torch.reshape(z, (b, n, self.num_heads, self.head_dims)).permute(0, 2, 1, 3), [q_, k_, v_])
        
        k_T = k.transpose(2, 3)
        attn_logits = torch.matmul(q, k_T) / (self.head_dims**(1/2))
        attn_weights = nn.Softmax(dim=-1)(attn_logits)
        attn_applied = torch.matmul(attn_weights, v).permute(0, 2, 1, 3)
        attn_applied = torch.reshape(attn_applied, (b, n, self.head_dims * self.num_heads))
        attn_out = self.W_O(attn_applied)
        
        return attn_out

class MLP(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims, bias=True):
        super().__init__()
        self.fc_1 = nn.Linear(input_dims, hidden_dims, bias=bias)
        self.fc_2 = nn.Linear(hidden_dims, output_dims, bias=bias)
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.1)
        
    def forward(self, x):
        o = F.gelu(self.fc_1(x))
        o = self.fc_2(o)
        return o

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dims, num_heads=4, bias=False):
        super(TransformerBlock, self).__init__()
        self.hidden_dims = hidden_dims
        self.norm1 = nn.LayerNorm(hidden_dims)
        self.self_attention = SelfAttention(hidden_dims, head_dims=hidden_dims // num_heads, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(hidden_dims)
        self.mlp = MLP(hidden_dims, hidden_dims, hidden_dims)
        
    def forward(self, x):
        skip = x
        out = self.norm1(x)
        out = self.self_attention(out)
        out += skip
        
        skip = out
        out = self.norm2(out)
        out = self.mlp(out)
        out += skip
        return out

class HRNetViT(nn.Module):
    def __init__(self, hidden_dims, input_dims=34, output_dims=1, num_trans_layers=4, num_heads=4, bias=False):
        super(PosT, self).__init__()
        self.hidden_dims = hidden_dims
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.num_trans_layers = num_trans_layers
        self.num_heads = num_heads
        self.sequence_length = 250
        
        self.linear_embed = nn.Linear(self.input_dims, self.hidden_dims)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.sequence_length + 1, self.hidden_dims))
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_dims))
        self.cls_index = torch.LongTensor([0])
        
        transformer_encoder_list = [TransformerBlock(self.hidden_dims, self.num_heads, bias) for _ in range(self.num_trans_layers)]
        self.transformer_encoder = nn.Sequential(*transformer_encoder_list)
        
        self.out_mlp = MLP(self.hidden_dims, self.hidden_dims, self.output_dims)
        
    def forward(self, x):
        b, n, d = x.shape
        x_embed = self.linear_embed(x)
        cls_tokens = self.cls_token.repeat(b, 1, 1)
        xcls_embed = torch.cat([cls_tokens, x_embed], dim=1)
        xcls_pos_embed = xcls_embed + self.pos_embedding[:, :n+1, :]
        trans_out = self.transformer_encoder(xcls_pos_embed)
        out_cls_token = trans_out[:, 0, :]
        out = self.out_mlp(out_cls_token)
        out = torch.sigmoid(out)
        return out.squeeze()
