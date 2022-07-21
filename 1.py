import torch
import torch.nn as nn

class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fn = nn.Linear(224,3)
        pass
    def forward(self, x):
        x = self.fn(x).contiguous()
        return x
    
device = torch.device('cuda:0')

x = torch.rand(3,3,224,224).to(device)
model = model()
model.to(device)
result = model(x)
print(result)
