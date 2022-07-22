import torch
import torch.nn as nn
import torch.optim as optim
class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 16, 16)  # [1, 14, 14]
        pass
    def forward(self, x, target):
        output = self.conv(x)
        loss = torch.sum(output - target)
        return output, loss
    
device = torch.device('cuda:0')

x = torch.rand(2, 3, 224, 224).to(device)
target = torch.rand(2, 1, 14, 14).to(device)
model = model()
model.to(device)
output, loss = model(x, target)
print(loss)
optimizer = torch.optim.SGD(model.parameters(),lr=1e-3,)
optimizer.zero_grad()
loss.backward()
optimizer.step()
