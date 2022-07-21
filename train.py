import os
import time
import copy
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim 
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
import torchaudio
from torchvision import transforms, datasets
from utils.dataset import img_dataset
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from models.modeling import VisionTransformer

def setup(config, pretrain = None):
    model = VisionTransformer(config)
    if pretrain:
        model.load_state_dict(torch.load(pretrain))
    return model
    
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
    return train_dataLoader, valid_dataLoader

def train(config, model, store_path):
    num_epochs = config['num_epochs']
    learining_rate = config['learining_rate']
    weight_decay = config['weight_decay']
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0
    val_acc_history = []
    train_acc_history = []
    train_losses = []
    valid_losses = []

    train_loader, valid_loader = creat_dataLoader(config['data_dir'],  config['batch_size'])

    device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=learining_rate,
                                momentum=0.9,
                                weight_decay=weight_decay)
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=num_epochs//20, t_total=num_epochs)
    # Train!
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        since = time.time()

        model.train()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = model(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicts = torch.max(outputs, dim=1)
            running_loss += loss.item() * inputs.size()
            running_corrects += torch.sum(predicts == labels)
            
        epoch_acc = running_corrects/len(train_loader.dataset)
        epoch_loss = running_loss/len(train_loader.dataset)

        time_elapsed = time.time() - since
        since = time.time()
        print('Time elapsed {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))



        pass
    

def main():
    CONFIG = {'img_size':224,
          'img_channel':3,
          'hidden_size':768,
          'patch_size':(16,16),
          'mlp_dim':3072,
          'num_layers':12,
          'num_heads':12,
          'drop_rate':0.1,
          'num_classes':3,
          'data_dir':'data',
          'batch_size':2,
          'num_epochs':200,
          'learning_rate':1e-2,
          'weight_decay':0,
    }

    

    model = setup(CONFIG)

    train(CONFIG, model, 'checkpoint')



if '__name__' == '__main__':
    main()

