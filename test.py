import time
import os
import sys

import torch
from dataloader.transformers import Rescale
from model.lanenet.LaneNet import LaneNet
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms
from model.utils.cli_helper_test import parse_args
import numpy as np
import os
from tqdm import tqdm
import time
from PIL import Image
import pandas as pd
import cv2

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_test_data(img_path, transform):
    img = Image.open(img_path)
    img = transform(img)
    return img


def predict():
    args = parse_args()
    img_fold = args.img_fold
    # 图片文件路径列表
    files = os.listdir(img_fold)
    # 生成预测图片的文件夹保存路径
    output_dir = args.save
    if os.path.exists(output_dir) == False:
        os.mkdir(output_dir)
    resize_height = args.height
    resize_width = args.width

    data_transform = transforms.Compose([
        transforms.Resize((resize_height,  resize_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    model_path = args.model
    model = LaneNet(arch=args.model_type)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)

    # 使用 tqdm 包裹 files，显示进度条
    with tqdm(total=len(files), desc="Processing files", unit="file", bar_format="{l_bar}{bar:40}{r_bar}") as pbar:
        for path in files:
            img_path = os.path.join(img_fold, path)

            dummy_input = load_test_data(img_path, data_transform).to(DEVICE)
            dummy_input = torch.unsqueeze(dummy_input, dim=0)
            outputs = model(dummy_input)

            input = Image.open(img_path)
            input = input.resize((resize_width, resize_height))
            input = np.array(input)

            instance_pred = torch.squeeze(outputs['instance_seg_logits'].detach().to('cpu')).numpy() * 255
            binary_pred = torch.squeeze(outputs['binary_seg_pred']).to('cpu').numpy() * 255

            cv2.imwrite(os.path.join(output_dir, path.split('.')[0] + '_input.jpg'), input)
            cv2.imwrite(os.path.join(output_dir, path.split('.')[0] + '_instance_output.jpg'), instance_pred.transpose((1, 2, 0)))
            cv2.imwrite(os.path.join(output_dir, path.split('.')[0] + '_binary_output.jpg'), binary_pred)

            # 更新进度条
            pbar.update(1)

    print('output is saved:', output_dir)


if __name__ == "__main__":
    predict()
