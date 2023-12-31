from __future__ import print_function

import csv

import torch
import numpy as np
from PIL import Image
import os
import cv2


def writetxt(name, value):
    with open('answer/ans.csv', mode='a+', newline='\n') as file:
        # 创建 CSV 写入对象
        writer = csv.writer(file)

        # 写入数据
        writer.writerows([[name, value]])


def tran(a):
    a1 = a.permute(0, 2, 3, 1)
    a2 = a1.cpu().numpy()[0]
    a3 = (a2 * 255).astype(int)
    return a3


def save_images(save_dir, visuals, image_name):
    """save images to disk"""
    image_name = image_name[0]
    palet_file = 'dataset/palette.txt'
    impalette = list(np.genfromtxt(palet_file, dtype=np.uint8).reshape(3 * 256))

    if visuals['ori_image'] != []:
        ori = visuals['ori_image']
        ori_img = tran(ori)
        ori_mask = tran(visuals['rgb_image'])

        green_mask = (ori_mask != 0).all(axis=2)  # 检查每个像素的所有通道是否都不为零
        ori_mask[green_mask] = [128, 0, 0]
        mask = ori_mask

    else:
        ori = visuals['rgb_image']
        ori_img = tran(ori)
        im_data = visuals['output']
        mask = tensor2labelim(im_data, impalette)

    h, w, _ = ori_img.shape
    non_zero_count = np.count_nonzero(mask)
    im1 = mask + ori_img
    value = round((non_zero_count / (h * w)) * 100)
    array = np.clip(im1, 0, 255)
    array = array.astype(np.uint8)
    cv2.imwrite(os.path.join(save_dir, image_name), cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
    writetxt(image_name,value)


def tensor2im(input_image, imtype=np.uint8):
    """Converts a image Tensor into an image array (numpy)"""
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    return image_numpy.astype(imtype)


def tensor2labelim(label_tensor, impalette, imtype=np.uint8):
    """Converts a label Tensor into an image array (numpy),
    we use a palette to color the label images"""
    if len(label_tensor.shape) == 4:
        _, label_tensor = torch.max(label_tensor.data.cpu(), 1)

    label_numpy = label_tensor[0].cpu().float().detach().numpy()
    label_image = Image.fromarray(label_numpy.astype(np.uint8))
    label_image = label_image.convert("P")
    label_image.putpalette(impalette)
    label_image = label_image.convert("RGB")
    return np.array(label_image).astype(imtype)


def print_current_losses(epoch, i, losses, t, t_data):
    message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, i, t, t_data)
    for k, v in losses.items():
        message += '%s: %.3f ' % (k, v)
    print(message)


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def confusion_matrix(x, y, n, ignore_label=None, mask=None):
    if mask is None:
        mask = np.ones_like(x) == 1
    k = (x >= 0) & (y < n) & (x != ignore_label) & (mask.astype(bool))
    return np.bincount(n * x[k].astype(int) + y[k], minlength=n ** 2).reshape(n, n)


def getScores(conf_matrix):
    if conf_matrix.sum() == 0:
        return 0, 0, 0, 0, 0
    with np.errstate(divide='ignore', invalid='ignore'):
        globalacc = np.diag(conf_matrix).sum() / conf_matrix.sum().astype(np.float32)
        classpre = np.diag(conf_matrix) / conf_matrix.sum(0).astype(np.float32)
        classrecall = np.diag(conf_matrix) / conf_matrix.sum(1).astype(np.float32)
        IU = np.diag(conf_matrix) / (conf_matrix.sum(1) + conf_matrix.sum(0) - np.diag(conf_matrix)).astype(np.float32)
        pre = classpre[1]
        recall = classrecall[1]
        iou = IU[1]
        F_score = 2 * (recall * pre) / (recall + pre)
    return globalacc, pre, recall, F_score, iou
