import os

import pandas as pd
from utils import get_files, get_dirs, get_dirs_train
import torch
import numpy as np


def train_load(img_path, label_path):
    label = pd.read_csv(label_path)
    types = ['Freeform']
    paths, labels = [], []
    for t in types:
        dirs = get_dirs_train(img_path + t)
        for d in dirs:
            p_path = os.listdir(d)
            for p in p_path:
                img_path = os.path.join(d, p)
                no = img_path.split('/')[-1]
                no = no[:5]
                l = label[label['file'] == no]['label'].to_numpy()[0]
                files = get_files(img_path)
                for file in files:
                    paths.append(d + '/' + p + '/' + file)
                    labels.append(l)
    return paths, torch.from_numpy(np.array(labels)).view((len(labels), 1))
