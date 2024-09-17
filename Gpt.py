import torch
import torch.nn as nn
import torch.nn.functional as F

from Transformer import Transformer_Block


class GPT2(nn.Module):
  def __init__(self, inp_dim, hid_dim, drop_prob, n_heads, n_trans, vocab_size):

    self.embedding = nn.Embedding(vocab_size, inp_dim) #Input Embedding [B, T, E]
    self.pos_embed = nn.Embedding(context_length, inp_dim) #Positional Embedding [B, T, E]
    #Input + Positional Embedding Sum [B, T, E]
    nn.ModuleList([Single_Head(inp_dim, hid_dim, drop_prob) for i in range(n_heads)])
    self.transformer_blocks = nn.Sequential(*[Transformer_Block(inp_dim, hid_dim, drop_prob, n_heads)  for i in range(n_trans)]) #n_trans number of Transformer Blocks [B, T, E] 
    self.layer_norm = nn.LayerNorm(inp_dim) #Layer Normalization [B, T, E]
    self.linear = nn.Linear(inp_dim, vocab_size) #Linear Layer [B, T, V] 
            
  def forward(self, x):
    B, T = x.shape
    
    x1 = self.embedding(x)  #Input Embedding [B, T, E]
    x2 = self.pos_embed(torch.arange(T, device=device))  #Positional Embedding [T, E]
    x = x1+x2 #Input + Positional Embedding Sum [B, T, E]
    x = self.transformer_blocks(x) #n_trans number of Transformer Blocks [B, T, E] 
    x = self.layer_norm(x) #Layer Normalization [B, T, E]
    x = self.linear(x) #Linear Layer [B, T, V] 

    return x