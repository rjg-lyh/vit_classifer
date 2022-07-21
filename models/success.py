import copy
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm



ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu}


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.num_attention_heads = 12
        self.attention_head_size = int(768 / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(768, self.all_head_size)
        self.key = Linear(768, self.all_head_size)
        self.value = Linear(768, self.all_head_size)

        self.out = Linear(768,768)
        self.attn_dropout = Dropout(0.1)
        self.proj_dropout = Dropout(0.1)

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        print(new_x_shape)
        x = x.view(*new_x_shape)
        print(x.shape)
        print(x.permute(0, 2, 1, 3).shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        print(hidden_states.shape)
        mixed_query_layer = self.query(hidden_states)
        print(mixed_query_layer.shape)
        mixed_key_layer = self.key(hidden_states)
        print(mixed_key_layer.shape)
        mixed_value_layer = self.value(hidden_states)
        print(mixed_value_layer.shape)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        print(query_layer.shape)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        print(key_layer.shape)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        print(value_layer.shape)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        print(attention_scores.shape)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        print(attention_scores.shape)
        attention_probs = self.softmax(attention_scores)
        print(attention_probs.shape)
        attention_probs = self.attn_dropout(attention_probs)
        print(attention_probs.shape)

        context_layer = torch.matmul(attention_probs, value_layer)
        print(context_layer.shape)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        print(context_layer.shape)
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        print(context_layer.shape)
        attention_output = self.out(context_layer)
        print(attention_output.shape)
        attention_output = self.proj_dropout(attention_output)
        print(attention_output.shape)
        return attention_output


class Mlp(nn.Module):
    def __init__(self):
        super(Mlp, self).__init__()
        self.fc1 = Linear(768, 12)
        self.fc2 = Linear(12,768)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(0.1)
        


    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self):
        super(Embeddings, self).__init__()
        patch_size = (16,16)
        n_patches = (224 // patch_size[0]) * (224 // patch_size[1])
       
        self.patch_embeddings = Conv2d(3,
                                       768,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, 768))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))

        self.dropout = Dropout(0.1)

    def forward(self, x):
        print(x.shape)
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        print(cls_tokens.shape)
        x = self.patch_embeddings(x)
        print(x.shape)
        x = x.flatten(2)
        print(x.shape)
        x = x.transpose(-1, -2)
        print(x.shape)
        x = torch.cat((cls_tokens, x), dim=1)
        print(x.shape)

        embeddings = x + self.position_embeddings
        print(embeddings.shape)
        embeddings = self.dropout(embeddings)
        print(embeddings.shape)
        return embeddings


class Block(nn.Module):
    def __init__(self):
        super(Block, self).__init__()
        self.hidden_size = 768
        self.attention_norm = LayerNorm(768, eps=1e-6)
        self.ffn_norm = LayerNorm(768, eps=1e-6)
        self.ffn = Mlp()
        self.attn = Attention()

    def forward(self, x):
        print(x.shape)
        h = x
        x = self.attention_norm(x)
        print(x.shape)
        x = self.attn(x)
        x = x + h
        print(x.shape)

        h = x
        x = self.ffn_norm(x)
        print(x.shape)
        x = self.ffn(x)
        print(x.shape)
        x = x + h
        print(x.shape)
        return x



class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(768, eps=1e-6)
        for _ in range(12):
            layer = Block()
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        print(hidden_states.shape)
        for layer_block in self.layer:
            hidden_states = layer_block(hidden_states)
        encoded = self.encoder_norm(hidden_states)
        return encoded


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings()
        self.encoder = Encoder()

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        encoded = self.encoder(embedding_output)
        return encoded


class VisionTransformer2(nn.Module):
    def __init__(self):
        super(VisionTransformer2, self).__init__()
        self.num_classes = 3
        self.transformer = Transformer()
        self.head = Linear(768,3)
        self.softmax = nn.Softmax(dim=1)
        self.CELoss = nn.CrossEntropyLoss()

    def forward(self, x, targets):
        x = self.transformer(x)
        print(x.shape)
        x = self.head(x[:, 0])
        print(x)
        targets = targets.t()
        loss = self.CELoss(self.softmax(x), targets)
        return x, loss
    
    # def loss_calcu(self, x, targets):
    #     targets = targets.t()
    #     x = self.softmax(x)
    #     loss = self.CELoss(x, targets)
    #     return loss



