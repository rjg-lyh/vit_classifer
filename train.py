import os
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

def train(config, model):
    epoch_num = config['epoch_num']
    train_dataLoader, valid_dataLoader = creat_dataLoader(config['data_dir'],  config['batch_size'])

    device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
    model.to(device)

    optimizer_ft = optim.Adam(model.parameters(), lr=1e-2)#要训练啥参数，你来定
    scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)#学习率每7个epoch衰减成原来的1/10
    criterion = nn.CrossEntropyLoss()

    

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
          'epoch_num':200,

    }

    

    model = setup(CONFIG)

    train(CONFIG, model)



if '__name__' == '__main__':
    main()

