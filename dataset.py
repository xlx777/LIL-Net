from __future__ import absolute_import

import numpy as np
from torch.utils.data import Dataset
from load_data import val_load, train_load
import torchvision
from PIL import Image

from torchvision.transforms import *
# from torchtoolbox.transform import Cutout
from PIL import Image
import random
import math
import numpy as np
import torch




class MyDataset(Dataset):
    def __init__(self, img_path, label_path, train=False):
        if train:
            self.path, self.label = train_load(img_path, label_path)
        else:
            self.path, self.label = train_load(img_path, label_path)
        self.train = train
        if self.train:
            # self.transform = torchvision.transforms.Compose([Normaliztion(),
            #                                                 RandomHorizontalFlip(),
            #                                                  ToTensor()])
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((256, 256)),
                torchvision.transforms.CenterCrop(224),
                # torchvision.transforms.RandomResizedCrop(224),
                # Cutout(),
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.RandomRotation(degrees=30),
                # torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
                torchvision.transforms.ToTensor(),
                # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                torchvision.transforms.Normalize(mean=[0.5833123, 0.4070842, 0.39774758],
                                                 std=[0.19454452, 0.19375803, 0.20291969]),
                # torchvision.transforms.RandomErasing()
            ])
        else:
            # self.transform = torchvision.transforms.Compose([Normaliztion(),
            #                                                 ToTensor()])
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((256, 256)),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                torchvision.transforms.Normalize(mean=[0.5833123, 0.4070842, 0.39774758],
                                                 std=[0.19454452, 0.19375803, 0.20291969]),
                # torchvision.transforms.Normalize(mean=[0.5], std=[0.5]),
            ])

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, idx):
        img = Image.open(self.path[idx])
        # img = np.float32(img) / 255
        img = self.transform(img)
        return img, self.label[idx], self.path[idx]

    def get_classes_for_all_imgs(self):
        import os
        path = '/home/dell/xlx/processed/17/processed/train/Freeform'
        score_path = os.listdir(path)
        # score_path.sort(key=lambda x: int(x.split('.')[0]))
        count = [0] * len(score_path)
        img_classes = []
        for i, score in enumerate(score_path):
            person_path = os.listdir(os.path.join(path, score))
            for p in person_path:
                # print(p)
                img_path = os.listdir(os.path.join(os.path.join(path, score), p))
                for img in img_path:
                    file_path = os.path.join(os.path.join(os.path.join(path, score), p), img)
                    img_classes.append(score)
                    count[i] = count[i] + 1
        # print(count)
        # print(img_classes)
        return list(map(int, img_classes)), list(map(int, score_path)), count
