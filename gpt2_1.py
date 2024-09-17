import torch
import torch.nn as nn
import torch.nn.functional as F




class Single_Head(nn.Module):
    def __init__(self,inp_dim,hid_dim,drop_prob):
        super(Single_Head,self).__init__()
        self.hid_dim=hid_dim 
        self.key=nn.Linear(inp_dim,hid_dim,bias=False)
        self.query=nn.Linear(inp_dim,hid_dim,bias=False)
        self.value=nn.Linear(inp_dim,hid_dim,bias=False)
        self.dropout=nn.Dropout(p=drop_prob)


    def forward(self,x):
        B, T, E=x.shape #Input [B, T, E],input_dim = E
        C = self.hid_dim # hidden dimension as C
        key = self.key(x) # Query [B, T, C] (Linear Map of Input)
        query = self.query(x) #Key [B, T, C] (Linear Map of Input)
        value = self.value(x) #value [B, T, C] (Linear Map of Input)

        matrix_mulp= query @ key.transpose(-2,-1)

        #Matrix Multiplication of Query and Key [B, T, T]
        normalize = matrix_mulp * (C **-0.5) #Nornalize the Matrix [B, T, T] with square root of C

        mask = torch.ones(T, T) #create a matrix T x T of ones
        mask = torch.tril(mask) #change the upper traingular part to zero

        masked_matrix = normalize.masked_fill(mask == 0, float('-inf')) #Masked Matrix for Parallel Computation [B, T, T]
        softmax = F.softmax(masked_matrix, dim = -1) # Softmax the Normalized [B, T, T]
        dropout = self.dropout(softmax) # Dropout [B, T, T]
        value_update = dropout @ value #Matrix Multiplication of Softmax by Vlaue [B, T, C]
        return value_update
    


class Multi_Head(nn.Module):

    def __init__(self, inp_dim, hid_dim, drop_prob, n_heads): #n_heads number of [B, T, C] inputs
        super(Multi_Head,self).__init__()

        self.hid_dim = hid_dim 
        self.attentionblocks = nn.ModuleList([Single_Head(inp_dim, hid_dim, drop_prob) for i in range(n_heads)]) #n_heads of parallel SIngle Attention Blocks
        self.linear = nn.Linear(hid_dim*n_heads,hid_dim*n_heads) # Linear Layer [B, T, C]


    def forward(self, x):
        x=torch.cat([h(x) for h in self.attentionblocks],dim = -1) # Concatenate n_heads outputs along the thrid dimenstion [B, T, C*n_heads]
        print(x.shape)
        x.self.linear(x) #Linear Layer [B, T, C]
        return x
    



class Transformer_Block(nn.Module):
  def __init__(self, inp_dim, hid_dim, drop_prob, n_heads):
    super(Transformer_Block,self).__init__()
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
  


batch_size = 32 #batchsize
context_length = 8 #context_length
D = 1 #
inp_dim = 64 #embedding dimension
hid_dim = 16 #hidden dimension / head size
n_heads = 4 # number of multi attention heads
# observe that inp_dim = hid_dim * n_heads
n_trans = 4 # number of transformer blocks
drop_prob = 0.6 #dropout probability
vocab_size = len(vocabulary)

class GPT2(nn.Module):
  def __init__(self, inp_dim, hid_dim, drop_prob, n_heads, n_trans, vocab_size):
    super(GPT2, self).__init__()

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

input = torch.randint(low=0, high=vocab_size, size=(batch_size, context_length))
model = GPT2(inp_dim, hid_dim, drop_prob, n_heads, n_trans, vocab_size)
model.eval()
output = model(input)
print(output.shape)