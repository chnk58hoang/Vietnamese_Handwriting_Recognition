import torch
from torch.nn.utils.rnn import pack_padded_sequence

x = torch.tensor([[1,0,0],[2,0,0],[1,1,1],[2,2,0]])

out = pack_padded_sequence(x,lengths=torch.tensor([1,1,3,2]),batch_first=True,enforce_sorted=False)
print(out.data)