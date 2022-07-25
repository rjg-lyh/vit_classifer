import os
import random 
from tqdm import tqdm
from torch import classes
import shutil

def copy_raw():
    root = '/home/rjg/project/data/raw_bearing/data'
    root1 = '/home/rjg/project/data/data_vit'
    root_train = '/home/rjg/project/data/data_vit/train'
    root_val = '/home/rjg/project/data/data_vit/valid'

    kinds = ['back','cover','endoscopy','none', 'open', 'seal'] 
    scope = {'back':[800,57], 'cover':[1350,600], 'endoscopy':[1700,1000], 'none':[1300,480], 'open':[1450,600], 'seal':[1600,1000]}
    for kind in kinds:
        new_path = os.path.join(root, kind)
        train_path = os.path.join(root_train, kind)
        val_path = os.path.join(root_val, kind)
        num1, num2 = 0, 0
        left, right = scope[kind][0], scope[kind][1]
        for name in tqdm(os.listdir(new_path)):
            img = os.path.join(new_path, name)
            img_train = os.path.join(train_path, name)
            img_valid = os.path.join(val_path, name)
            if num1 >= left and num2 >= right:
                break
            if num1 < left and num2 < right:
                a = random.random()
                if a < 0.5:
                    shutil.copy(img, img_train)
                    num1 += 1
                else:
                    shutil.copy(img, img_valid)
                    num2 += 1
                continue
            if num2 >= right:
                shutil.copy(img, img_train)
                num1 += 1
                continue
            shutil.copy(img, img_valid)
            num2 += 1
    return root1

if __name__ == '__main__':
    copy_raw()
        


