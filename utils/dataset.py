import torch
from cProfile import label
import json
from PIL import Image
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader

class img_dataset(DataLoader):
    def __init__(self, image_root, label_place, transform):
        self.pocessed = self.split_dataset(label_place)
        self.images = [os.path.join(image_root, path) for path in self.pocessed.keys()]
        self.labels = list(self.pocessed.values())
        self.transform = transform
        
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        label = torch.from_numpy(self.labels[idx])
        if self.transform:
            image = self.transform(image)
        return image, label

    def split_dataset(self, label_place):
        pocessed = {}
        with open(label_place, 'r') as f:
            for s in f.readlines():
                path, label = s.strip().split(' ') 
                pocessed[path] = np.array(label, dtype=np.int64)
        return pocessed



