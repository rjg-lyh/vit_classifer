from turtle import forward
from boto import config
import torch
import torch.nn as nn
import copy

class Embedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        img_size = config["img_size"]
        img_channel = config["img_channel"]
        patch_size = config["patch_size"][0]
        patch_n = (img_size // patch_size)**2
        hidden_size = config["hidden_size"]
        self.dropout = nn.Dropout(config['drop_rate'])   
        self.patch_construct = nn.Conv2d(img_channel, hidden_size, patch_size, patch_size) 
        self.class_patch = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.location = nn.Parameter(torch.zeros(1, patch_n + 1, hidden_size))
        pass
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_construct(x) #[2, 768, 14, 14]
        x = x.flatten(2) #[2, 768, 196]
        x = x.transpose(2,1) #[2, 196, 768]
        class_patch = self.class_patch.expand(B, -1, -1)   # !!!!!!注意不能写成self.class_patch = self.class_patch.expand(B, -1, -1)
                                                             # 因为parameter是可训练的
        x = torch.cat((x, class_patch),dim=1)  #[2, 197, 768]
        x = x + self.location
        x = self.dropout(x)
        return x

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config['hidden_size']
        self.num_heads = config['num_heads']
        self.head_size = self.hidden_size // self.num_heads
        self.query = nn.Linear(self.hidden_size, self.hidden_size)
        self.key = nn.Linear(self.hidden_size, self.hidden_size)
        self.value = nn.Linear(self.hidden_size, self.hidden_size)
        pass
    def transpose_multiHeads(self, x):
        new_shape = x.size()[:2] + (self.num_heads, self.head_size)
        x = x.view(*new_shape).permute(0, 2, 1, 3)
        return x

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        multi_query = self.transpose_multiHeads(query)#[3, 12, 197, 64]
        multi_key = self.transpose_multiHeads(key)
        multi_value = self.transpose_multiHeads(value)
        multi_attention = torch.matmul(multi_query, multi_value.transpose(-1,-2)) #[3, 12, 197, 197]
        x = torch.matmul(multi_attention, multi_value) ##[3, 12, 197, 64]
        x = x.permute(0, 2, 1, 3).contiguous()
        new_shape = x.size()[:2] + (self.hidden_size, )
        x = x.view(*new_shape) #[3, 197, 768]
        return x

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config['hidden_size']
        mlp_dim = config['mlp_dim']
        drop_rate = config['drop_rate']
        self.fn1 = nn.Linear(hidden_size, mlp_dim)
        self.fn2 = nn.Linear(mlp_dim, hidden_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(drop_rate)
        pass
    def forward(self, x):
        x = self.fn1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fn2(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln = nn.LayerNorm(config['hidden_size'], eps=1e-6)
        self.attn = Attention(config)
        self.fn = MLP(config)

    def forward(self, x):
        h = x
        x = self.ln(x)
        x = self.attn(x)
        x += h
        h = x
        x = self.ln(x)
        x = self.fn(x)
        x += h
        return x

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        num_layers = config['num_layers']
        self.block = Block(config)
        self.ln = nn.LayerNorm(config['hidden_size'], eps=1e-6)
        self.layer_list = nn.ModuleList()
        for _ in range(num_layers):
            self.layer_list.append(copy.deepcopy(self.block))
        
    def forward(self, x):
        for block in self.layer_list:   
            x = block(x)              
        x = self.ln(x)
        return x

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = Embedding(config)
        self.encoder = Encoder(config)
        pass
    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config['hidden_size']
        num_classes = config['num_classes']
        self.transformer = Transformer(config)
        self.head = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.CELoss = nn.CrossEntropyLoss()
        pass
    def forward(self, x, targets):
        x = self.transformer(x)
        x = self.head(x[:, 0])
        targets = targets.t()
        loss = self.CELoss(self.softmax(x), targets)
        return x, loss

