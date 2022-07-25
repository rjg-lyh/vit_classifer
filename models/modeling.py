from turtle import forward
from boto import config
import logging
import torch
import torch.nn as nn
import numpy as np
from os.path import join as pjoin
from scipy import ndimage
import copy
import math

logger = logging.getLogger(__name__)

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

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
        #print(B)
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
        self.drop_rate = config['drop_rate']
        self.head_size = self.hidden_size // self.num_heads
        self.query = nn.Linear(self.hidden_size, self.hidden_size)
        self.key = nn.Linear(self.hidden_size, self.hidden_size)
        self.value = nn.Linear(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.hidden_size)

        self.softmax = nn.Softmax(dim=-1)
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
        multi_attention = multi_attention / math.sqrt(self.head_size)

        multi_attention_probs = self.softmax(multi_attention)
        x = torch.matmul(multi_attention_probs, multi_value) ##[3, 12, 197, 64]
        x = x.permute(0, 2, 1, 3).contiguous()
        new_shape = x.size()[:2] + (self.hidden_size, )
        x = x.view(*new_shape) #[3, 197, 768]
        x = self.out(x)
        return x

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config['hidden_size']
        mlp_dim = config['mlp_dim']
        drop_rate = config['drop_rate']
        self.fc1 = nn.Linear(hidden_size, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, hidden_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(drop_rate)
        pass
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config['hidden_size']
        self.attention_norm = nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.attn = Attention(config)
        self.ffn = MLP(config)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x = self.attn(x)
        x += h
        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x += h
        return x
    
    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():   
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))
            

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
        self.embeddings = Embedding(config)
        self.encoder = Encoder(config)
        pass
    def forward(self, x):
        x = self.embeddings(x)
        x = self.encoder(x)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config['hidden_size']
        num_classes = config['num_classes']
        self.classifier = "token"
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

    def load_from(self, weights):
        with torch.no_grad():
            # self.head.weight.copy_(np2th(weights["head/kernel"]).t())
            # self.head.bias.copy_(np2th(weights["head/bias"]).t())

            self.transformer.embeddings.patch_construct.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_construct.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer.embeddings.class_patch.copy_(np2th(weights["cls"]))
            self.transformer.encoder.ln.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.ln.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.location
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.location.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.location.copy_(np2th(posemb))
            
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    if hasattr(unit, 'load_from'):
                        unit.load_from(weights, n_block=uname)
            print('load the pretrain_model successfully'.center(100,'-'))