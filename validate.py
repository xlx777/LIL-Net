import os
import time

import torch


def validate(model, test_loader, device, criterion1, criterion2):
    model.eval()
    with torch.no_grad():
        rmse, mae, loss_all = 0., 0., 0
        val_rmes, val_mae, val_loss = 0., 0., 0.
        step = 0
        loca = time.strftime('%Y-%m-%d-%H-%M-%S')
        for img, label, _ in test_loader:
            img, label = img.to(device), label.to(device).to(torch.float32)
            score = model(img)
            # score = torch.transpose(score, 0, 1)
            s_mean = torch.mean(score)
            l_mean = torch.mean(label)
            loss = criterion2(score, label)
            loss_all += loss.item()
            rmse = torch.sqrt(torch.pow(torch.abs(score - label), 2).mean()).item()
            mae = torch.abs(score - label).mean().item()
            val_rmes += rmse
            # mae = torch.abs(score - label).mean().item()
            val_mae += mae
            step_name = str(loca) + "-validate.txt"
            s = 'Step:{} \tTrain RMSE:{:.2f} MAE:{:.2f} \t score:{:.2f} \t label:{:.2f}\t' \
                .format(step, rmse, mae, s_mean.item(),
                        l_mean.item())
            with open(os.path.join('result', step_name), 'a', encoding='utf-8') as f:
                f.write(s)
                f.write('\n')
            step += 1
        val_rmes /= step
        val_mae /= step
        loss_all /= step
    return val_rmes, val_mae, loss_all
