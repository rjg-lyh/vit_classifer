import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy

# weights = np.load('ViT-B_16.npz')
# for name, param in weights.items():
#     print(name, param.shape)
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.fn = nn.Linear(6,6)
    def forward(self):
        pass

class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(6, 1e-3)
        self.layer = nn.ModuleList()
        for _ in range(3):
            block = Block()
            self.layer.append(copy.deepcopy(block))
    def forward(self):
        pass

model = model()
for name, module in model.named_children():
    print(name, module)
    print('下面是子模块'.center(100,'-'))
    for name1, module1 in module.named_children():
        print(name1)
