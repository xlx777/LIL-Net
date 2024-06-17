from __future__ import absolute_import

from torch.utils.data import Dataset
from load_data import train_load
import torchvision
from PIL import Image


class MyDataset(Dataset):
    def __init__(self, img_path, label_path, train=False):
        if train:
            self.path, self.label = train_load(img_path, label_path)
        else:
            self.path, self.label = train_load(img_path, label_path)
        self.train = train
        if self.train:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((256, 256)),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.RandomRotation(degrees=30),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.5833123, 0.4070842, 0.39774758],
                                                 std=[0.19454452, 0.19375803, 0.20291969]),
            ])
        else:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((256, 256)),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.5833123, 0.4070842, 0.39774758],
                                                 std=[0.19454452, 0.19375803, 0.20291969]),
            ])

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, idx):
        img = Image.open(self.path[idx])
        img = self.transform(img)
        return img, self.label[idx], self.path[idx]

    def get_classes_for_all_imgs(self):
        import os
        path = 'path'
        score_path = os.listdir(path)
        count = [0] * len(score_path)
        img_classes = []
        for i, score in enumerate(score_path):
            person_path = os.listdir(os.path.join(path, score))
            for p in person_path:
                img_path = os.listdir(os.path.join(os.path.join(path, score), p))
                for img in img_path:
                    img_classes.append(score)
                    count[i] = count[i] + 1
        return list(map(int, img_classes)), list(map(int, score_path)), count
