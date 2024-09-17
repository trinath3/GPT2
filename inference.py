import torch 
 

from GPT import Gpt2



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

input = torch.randint(low=0, high=vocab_size, size=(batch_size, context_length))
model = GPT2(inp_dim, hid_dim, drop_prob, n_heads, n_trans, vocab_size)
model.eval()
output = model(input)
print(output.shape)