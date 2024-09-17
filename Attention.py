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
        self.hid_dim = hid_dim 
        self.attentionblocks = nn.ModuleList([Single_Head(inp_dim, hid_dim, drop_prob) for i in range(n_heads)]) #n_heads of parallel SIngle Attention Blocks
        self.linear = nn.Linear(hid_dim*n_heads,hid_dim*n_heads) # Linear Layer [B, T, C]


    def forward(self, x):
        x=torch.cat([h(x) for h in self.attentionblocks],dim = -1) # Concatenate n_heads outputs along the thrid dimenstion [B, T, C*n_heads]
        print(x.shape)
        x.self.linear(x) #Linear Layer [B, T, C]
        return x