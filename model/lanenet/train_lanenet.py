import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import os
import numpy as np
import time
from tqdm import tqdm
import copy
from matplotlib import pyplot as plt
import scipy.signal
from model.lanenet.loss import DiscriminativeLoss, FocalLoss


def drawing_loss(log_dir, train_loss, val_loss):
    # 绘制loss的曲线
    iters = range(1, len(train_loss) + 1)

    # 创建画布
    plt.figure()
    # 绘制loss和val_loss
    plt.plot(iters, train_loss, 'red', linewidth=2, label='train loss')
    plt.plot(iters, val_loss, 'blue', linewidth=2, label='val loss')

    # 绘制其他的一些细节
    plt.grid(True)  # 是否带背景网格
    plt.xlabel('Epoch')  # x轴变量名称
    plt.ylabel('Loss')  # y轴变量名称
    plt.legend(loc="upper right")  # 在右上角绘制图例标签

    # 保存图片到所在路径
    plt.savefig(os.path.join(log_dir, "epoch_loss.png"))

    plt.cla()  # 清除axes，即当前 figure 中的活动的axes，但其他axes保持不变
    plt.close("all")  # 关闭 window，如果没有指定，则指当前 window。


def compute_loss(net_output, binary_label, instance_label, loss_type='FocalLoss'):
    k_binary = 10  # 1.7
    k_instance = 0.3
    k_dist = 1.0

    if loss_type == 'FocalLoss':
        loss_fn = FocalLoss(gamma=2, alpha=[0.25, 0.75])
    elif loss_type == 'CrossEntropyLoss':
        loss_fn = nn.CrossEntropyLoss()
    else:
        # print("Wrong loss type, will use the default CrossEntropyLoss")
        loss_fn = nn.CrossEntropyLoss()

    binary_seg_logits = net_output["binary_seg_logits"]
    binary_loss = loss_fn(binary_seg_logits, binary_label)

    pix_embedding = net_output["instance_seg_logits"]
    ds_loss_fn = DiscriminativeLoss(0.5, 1.5, 1.0, 1.0, 0.001)
    var_loss, dist_loss, reg_loss = ds_loss_fn(pix_embedding, instance_label)
    binary_loss = binary_loss * k_binary
    var_loss = var_loss * k_instance
    dist_loss = dist_loss * k_dist
    instance_loss = var_loss + dist_loss
    total_loss = binary_loss + instance_loss
    out = net_output["binary_seg_pred"]

    return total_loss, binary_loss, instance_loss, out


def train_model(model, optimizer, save_path, scheduler, dataloaders, dataset_sizes, device,
                loss_type='FocalLoss', num_epochs=25):
    since = time.time()
    training_log = {'epoch': [], 'training_loss': [], 'val_loss': []}

    # 设置默认最小loss
    best_loss = float("inf")

    best_model_wts = copy.deepcopy(model.state_dict())

    # -----------开始训练-------------
    for epoch in range(num_epochs):
        training_log['epoch'].append(epoch)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_loss_b = 0.0
            running_loss_i = 0.0

            # Iterate over data.

            # 添加进度条
            iterations = len(dataloaders[phase])
            if phase == 'train':
                print('Start Train...')
                print('\n{:^15}{:^15}{:^15}{:^15}'.format('Epoch', 'Total Loss', 'Binary Loss', 'Instance Loss'))
            if phase == 'val':
                print('Start validation...')
                print('\n{:^15}{:^15}'.format('Epoch', 'Loss'))
            with tqdm(total=iterations) as pbar_train:
                for batch_idx, batch in enumerate(dataloaders[phase]):
                    inputs, binarys, instances = batch
                    inputs = inputs.type(torch.FloatTensor).to(device)
                    binarys = binarys.type(torch.LongTensor).to(device)
                    instances = instances.type(torch.FloatTensor).to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = compute_loss(outputs, binarys, instances, loss_type)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss[0].backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss[0].item() * inputs.size(0)
                    running_loss_b += loss[1].item() * inputs.size(0)
                    running_loss_i += loss[2].item() * inputs.size(0)

                    # 更新进度条
                    epoch_loss = running_loss / (batch_idx + 1)
                    binary_loss = running_loss_b / (batch_idx + 1)
                    instance_loss = running_loss_i / (batch_idx + 1)

                    if phase == 'train':
                        pbar_train.set_description('{:^15}{:^15.4f}{:^15.4f}{:^15.4}'.format(
                            f'{epoch + 1}/{num_epochs}', epoch_loss, binary_loss, instance_loss))
                        training_log['training_loss'].append(epoch_loss)

                    if phase == 'val':
                        pbar_train.set_description('{:^15}{:^15.4f}'.format(
                            f'{epoch + 1}/{num_epochs}', epoch_loss))
                        training_log['val_loss'].append(epoch_loss)

                    pbar_train.update(1)

                if phase == 'train':
                    if scheduler is not None:
                        scheduler.step()

                    # 保存last model 与 best model
                    if epoch_loss < best_loss:
                        best_loss = epoch_loss
                        best_model_wts = copy.deepcopy(model.state_dict())
                        torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))

                    torch.save(model.state_dict(), os.path.join(save_path, 'last_model.pth'))
                    # print("model is saved: {}".format(save_path))

    # loss绘制
    drawing_loss(save_path, training_log['training_loss'], training_log['val_loss'])

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val_loss: {:4f}'.format(best_loss))
    training_log['training_loss'] = np.array(training_log['training_loss'])
    training_log['val_loss'] = np.array(training_log['val_loss'])

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, training_log


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable
