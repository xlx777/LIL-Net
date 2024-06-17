import os
import time
import torch
from dataset import MyDataset
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from models.LIL import pyramid_trans_expr2

batch_size = 48
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = pyramid_trans_expr2(img_size=224, num_classes=1).to(device)
model_dict = torch.load('xx.pth', map_location=device)
model.load_state_dict(model_dict)
dataset_test = MyDataset('test_path', 'label_path')
num_test = len(dataset_test)
test_loader = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False, pin_memory=True,
                         drop_last=False)
test_rmse, test_mae = 0., 0.
rmse, mae = 0., 0.
step = 0
paths, labels, predicts = [], [], []
model.eval()
with torch.no_grad():
    loader = tqdm(test_loader)
    loca = time.strftime('%Y-%m-%d-%H-%M-%S')
    for img, label, path in loader:
        paths += list(path)
        labels += torch.flatten(label).tolist()
        img, label = img.to(device), label.to(device).to(torch.float32)
        predict = model(img)
        s_mean = torch.mean(predict)
        l_mean = torch.mean(label)
        predicts += torch.flatten(predict).tolist()
        rmse = torch.sqrt(torch.pow(torch.abs(predict - label), 2).mean()).item()
        mae = torch.abs(predict - label).mean().item()
        test_rmse += rmse
        test_mae += mae
        step_name = str(loca) + "-test.txt"
        s = 'Step:{} \tTrain RMSE:{:.2f} MAE:{:.2f} \t score:{:.2f} \t label:{:.2f}\t' \
            .format(step, rmse, mae, s_mean.item(),
                    l_mean.item())
        with open(os.path.join('result', step_name), 'a', encoding='utf-8') as f:
            f.write(s)
            f.write('\n')
        step += 1
        loader.set_description('step:{} {}/{}'.format(step, step * batch_size, num_test))
    test_rmse /= step
    test_mae /= step
print('Test\tMAE:{}\t RMSE:{}'.format(test_mae, test_rmse))
pd.DataFrame({'file': paths, 'label': labels, 'predict': predicts}).to_csv('testInfo.csv', index=False)
