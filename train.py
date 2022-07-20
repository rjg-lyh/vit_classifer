import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from utiles.dataset import img_dataset
from models.modeling import VisionTransformer

CONFIG = {'img_size':224,
          'img_channel':3,
          'hidden_size':768,
          'patch_size':(16,16),
          'mlp_dim':3072,
          'num_layers':12,
          'num_heads':12,
          'drop_rate':0.1,
          'num_classes':3
}

image_train_root = 'data/train'
image_valid_root = 'data/valid'
train_place = 'train.txt'
valid_place = 'valid.txt'

batch_size = 2

data_transforms = {
    'train': 
        transforms.Compose([
        transforms.Resize(64),
        transforms.RandomRotation(45),#随机旋转，-45到45度之间随机选
        transforms.CenterCrop(64),#从中心开始裁剪
        transforms.RandomHorizontalFlip(p=0.5),#随机水平翻转 选择一个概率概率
        transforms.RandomVerticalFlip(p=0.5),#随机垂直翻转
        transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),#参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
        transforms.RandomGrayscale(p=0.025),#概率转换成灰度率，3通道就是R=G=B
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])#均值，标准差
    ]),
    'valid': 
        transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

#or：  datasets.ImageFolder(data_dir, data_transforms)

train_dataset = img_dataset(image_train_root, train_place, data_transforms['train'])
valid_dataset = img_dataset(image_valid_root, valid_place, data_transforms['valid'])

train_dataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataLoader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

model = VisionTransformer(CONFIG)

# x = torch.randn(2, 3, 224, 224)
# print(model(x))