import os
import time

import torch
import torch.nn as nn
from validate import validate
from tqdm import tqdm
from models.LIL import pyramid_trans_expr2


def train(train_loader, test_loader, epochs, lr, device, model_dict):
    best_l = 1000
    best_epoch = -1
    model = pyramid_trans_expr2(img_size=224, num_classes=1).to(device)
    # model = GCViT(depths=[3, 4, 19, 5],
    #                      num_heads=[3, 6, 12, 24],
    #                      window_size=[7, 7, 14, 7],
    #                      dim=96,
    #                      mlp_ratio=2,
    #                      layer_scale=1e-5,
    #                      num_classes=1,
    #                      drop_path_rate=0.1).to(device)
    optimizer_e = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_e, mode='min', patience=5, factor=0.5,
                                                              min_lr=1e-7)
    criterion1 = nn.MSELoss()
    criterion2 = nn.L1Loss()
    loca = time.strftime('%Y-%m-%d-%H-%M-%S')
    for epoch in range(epochs):
        local2 = time.strftime('%Y-%m-%d-%H-%M-%S')
        if epoch == 0:
            total = sum([param.nelement() for param in model.parameters()])
            print("Number of parameter: %.2fM" % (total / 1e6))
        model.train()
        train_rmes, train_mae, train_loss = 0., 0., 0.
        step = 0
        loader = tqdm(train_loader)
        for img, label, _ in loader:
            img, label = img.to(device), label.to(device).to(torch.float32)
            optimizer_e.zero_grad()
            score = model.forward(img)
            # score = torch.transpose(score, 0, 1)
            s_mean = torch.mean(score)
            l_mean = torch.mean(label)
            # reg_loss = l2_regularization(model, 1e-1).to(device)
            loss = criterion2(score, label)
            train_loss += loss.item()
            # rmse1 = torch.pow((label - score), 2)
            rmse = torch.sqrt(torch.pow(torch.abs(label - score), 2).mean()).item()
            train_rmes += rmse
            mae = torch.abs(label - score).mean().item()
            train_mae += mae
            loss.backward()
            optimizer_e.step()
            step_name = str(local2) + "step.txt"
            s = 'Step:{} \t Train RMSE:{:.2f} MAE:{:.2f} \t loss:{:.2f} \t score:{:.2f} \t label:{:.2f}\t'.format(
                step,
                rmse,
                mae,
                loss,
                s_mean.item(),
                l_mean.item())
            with open(os.path.join('result', step_name), 'a', encoding='utf-8') as f:
                f.write(s)
                f.write('\n')
            step += 1
            loader.set_description("Epoch:{} Step:{} RMSE:{:.2f} MAE:{:.2f}".format(epoch, step, rmse, mae))
        train_rmes /= step
        train_mae /= step
        train_loss /= step
        # model.eval()
        val_rmes, val_mae, val_loss = validate(model, test_loader, device, criterion1, criterion2)
        lr_scheduler.step(val_loss)
        if val_loss < best_l:
            print('Save model!,Loss Improve:{:.2f}'.format(best_l - val_loss))
            best_l = val_loss
            best_epoch = epoch
            if not os.path.exists(model_dict + '/' + loca):
                os.makedirs(model_dict + '/' + loca)
            torch.save(model.state_dict(), model_dict + '/' + loca + '/' + 'Poster' + '_{:.2f}.pth'.format(val_loss))
        print('Train lr:{:.0e} \n'
              'Train RMSE:{:.2f} MAE:{:.2f} \n'
              'Val RMSE:{:.2f} MAE:{:.2f} \n'
              'Best epoch:{:.0f}\n'
              'Best test loss:{:.2f}'.format(optimizer_e.param_groups[0]['lr'], train_rmes, train_mae, val_rmes,
                                             val_mae, best_epoch, best_l))
