from cProfile import label
import os
import time
import copy
from tables import Col
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim 
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
import torchaudio
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from utils.dataset import img_dataset
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from models.modeling import VisionTransformer

def creat_dataLoader(data_dir, batch_size):
    data_transforms = {
    'train': 
        transforms.Compose([
        transforms.Resize(224),
        transforms.RandomRotation(45),#随机旋转，-45到45度之间随机选
        transforms.CenterCrop(224),#从中心开始裁剪
        transforms.RandomHorizontalFlip(p=0.5),#随机水平翻转 选择一个概率概率
        transforms.RandomVerticalFlip(p=0.5),#随机垂直翻转
        transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),#参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
        transforms.RandomGrayscale(p=0.025),#概率转换成灰度率，3通道就是R=G=B
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])#均值，标准差
    ]),
    'valid': 
        transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    }
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid']}
    # train_dataset = img_dataset(image_train_root, train_place, data_transforms['train'])
    # valid_dataset = img_dataset(image_valid_root, valid_place, data_transforms['valid'])

    train_dataLoader = DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True)
    valid_dataLoader = DataLoader(image_datasets['valid'], batch_size=batch_size, shuffle=True)

    # print(image_datasets['train'].classes)  #根据分的文件夹的名字来确定的类别
    # print(image_datasets['train'].class_to_idx) #按顺序为这些类别定义索引为0,1...
    # print(image_datasets['train'].imgs) #返回从所有文件夹中得到的图片的路径以及其类别
    return image_datasets, train_dataLoader, valid_dataLoader

image_datasets, train, valid = creat_dataLoader('data',2)
train_dataset1, train_dataset2 = torch.utils.data.random_split(image_datasets['train'], [2, 4])
mm = train_dataset1 + train_dataset2
#print('...........',mm.classes) 无效
print(type(mm))
print(type(image_datasets['train'] + image_datasets['valid']))
train_dataLoader = DataLoader(train_dataset1, batch_size=1, shuffle=True)
for inputs, labels in train_dataLoader:
    print(inputs.shape)
    print(labels.shape)
# print(type(image_datasets['train']))
# print(train_dataset2)
# print(len(train_dataset1 + train_dataset2))
# print(len(train))
# print(len(valid))