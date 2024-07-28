#teacher feature shape : torch.Size([16, 1280, 4, 4])
# student feature shape : torch.Size([16, 1280, 4, 4])

import torch
import math

max_seq_length = 16
position = torch.arange(max_seq_length).unsqueeze(1) # [16,1]


embed_dim = 340
div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)) # [frame_num, dim]
print(f'div_term shape : {div_term.shape}') # [170]

# [1] make positional encoding
pe = torch.zeros(1, max_seq_length, embed_dim) # [1,16,340]
div_pos = position * div_term
print(f'div_pos shape : {div_pos.shape}') # [16,170]
pe[:,0::2,0] = torch.sin(position * div_term) # [16,1] * [170] = [16,170] # odd number = sin
pe[:,1::2,0] = torch.cos(position * div_term) # [16,1] * [170] = [16,170] # odd number = sin
"""
pe[0, :, 0::2] = torch.sin(position * div_term) # [16,1] * [170] = [16,170] # odd number = sin

pe[0, :, 1::2] = torch.cos(position * div_term)

div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)) # [170] position info
"""
#
"""
position = torch.arange(self.max_seq_length).unsqueeze(1) # [len,1]

        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)) # [
        pe = torch.zeros(1, self.max_seq_length, embed_dim) # 0,
        pe[0, :, 0::2] = torch.sin(position * div_term) # interpolate ?
        pe[0, :, 1::2] = torch.cos(position * div_term)
"""
