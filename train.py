import os
import sys
import cv2
import shutil
import warnings
from model import SANet
from utils import SSIM_Loss, weights_normal_init
from data import MyDataset

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader


def main():
    model = SANet().cuda() if torch.cuda.is_available() else SANet()
    weights_normal_init(model)
    img_file_path = os.path.join('StoneData', 'images')
    gt_file_path = os.path.join('StoneData', 'ground_truth')
    train_loader = DataLoader(MyDataset(img_file_path, gt_file_path), batch_size=1, shuffle=True, num_workers=4)
    criterion = nn.MSELoss(size_average=False).cuda() if torch.cuda.is_available() else nn.MSELoss(size_average=False)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    train(model, train_loader, criterion, optimizer, 0)

def train(model, train_loader, criterion, optimizer, epoch):
    cuda_stat = torch.cuda.is_available()
    step_num = 0
    total_loss = 0.0
    step_loss = 0.0
    for i, (img, gt) in enumerate(train_loader):
        img = Variable(img.cuda()) if cuda_stat else Variable(img)
        gt = Variable(gt.cuda()) if cuda_stat else Variable(gt)
        out_put = model(img)
        loss = criterion(out_put, gt)
        step_loss = loss.item()
        total_loss += step_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step_num = step_num + 1
        print("Epoch %d: step %d ends. loss is %.4f" % (epoch, step_num, step_loss))
    print("Epoch %d ended: average loss is %.4f" % (epoch, total_loss / step_num))



if __name__ == '__main__':
    main()