import random

import numpy as np

import os
from dataset import MyDataset
from torch.utils.data import DataLoader, WeightedRandomSampler
from train import train
import torch
seed = 627
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
batch_size = 48
lr = 1e-4
epochs = 200
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_dir = './model_dict'
log_dir = './log'
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
dataset_train = MyDataset('train_path', 'label_path',
                          train=True)
train_targets, score_path, class_sample_counts = dataset_train.get_classes_for_all_imgs()
weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)
samples_weights = []
for t in train_targets:
    if t in score_path:
        samples_weights.append(weights[score_path.index(t)])
samples_weights = np.array(samples_weights)
sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)
train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=False, pin_memory=True,
                          drop_last=True, sampler=sampler, num_workers=4)

dataset_test = MyDataset('dev_path', 'label_path')
test_loader = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=4)

train(train_loader, test_loader, epochs, lr, device, model_dir)
