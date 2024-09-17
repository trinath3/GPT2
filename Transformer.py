import torch
import torch.nn as nn
import torch.nn.functional as F
from Attention import Multi_Head

class Transformer_Block(nn.Module):
  def __init__(self, inp_dim, hid_dim, drop_prob, n_heads):
    #n_embd = 
    self.layer_norm1 = nn.LayerNorm(inp_dim) #Layer Normalization [B, T, C]
    self.multi_head = Multi_Head(inp_dim, hid_dim, drop_prob, n_heads) #Multi-Head Attention Block [B, T, C']
    self.dropout1 = nn.Dropout(drop_prob) #Dropout [B, T, C']
    #skip connection here
    self.layer_norm2 = nn.LayerNorm(hid_dim*n_heads) #Layer Normalization [B, T, C']
    self.linear1 = nn.Linear(hid_dim*n_heads, 4*hid_dim*n_heads) #Linear Layer [B, T, 4C']
    self.gelu = nn.GELU() #GeLu [B, T, 4C']
    self.linear2 = nn.Linear(4*hid_dim*n_heads, hid_dim*n_heads) #Linear Layer [B, T, 4C']
    self.dropout2 = nn.Dropout(drop_prob) #Dropout [B, T, C']
    #skip connection here

    #skip connnections will be there in the feedforward network part

  def forward(self, x):
    x1 = self.layer_norm1(x) #Layer Normalization [B, T, C]
    x1 = self.multi_head(x1) #Multi-Head Attention Block [B, T, C']
    x1 = self.dropout1(x1) #Dropout [B, T, C']
    x1 = x + x1 #Skip Connection [B, T, C']
    x1 = self.layer_norm2(x1) #Layer Normalization [B, T, C']
    x2 = self.linear1(x1) #Linear Layer [B, T, 4C']
    x2 = self.gelu(x2) #GeLu [B, T, 4C']
    x2 = self.linear2(x2) #Linear Layer [B, T, 4C']
    x2 = self.dropout2(x2) #Dropout [B, T, C']
    x3 = x1 + x2 #Skip Connection [B, T, C']

    return x3