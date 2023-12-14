import os.path
import torchvision.transforms as transforms
import torch
import cv2
import numpy as np
from data.base_dataset import BaseDataset
import torch.nn.functional as F
import albumentations as A
import matplotlib.pyplot as plt
import random

def img_label(label_image):
    label = np.zeros((label_image.shape[0], label_image.shape[1]), dtype=np.uint8)
    label[label_image[:, :, 0] > 0] = 1
    label = torch.from_numpy(label)
    label = label.type(torch.LongTensor)
    return label

def rotate(rgb_image, anglex, angley, anglez):
    # 定义欧拉角（角度）
    roll_angle = anglex  # 绕 X 轴旋转
    pitch_angle = angley  # 绕 Y 轴旋转
    yaw_angle = anglez  # 绕 Z 轴旋转

    # 将角度转换为弧度
    roll_angle_rad = np.radians(roll_angle)
    pitch_angle_rad = np.radians(pitch_angle)
    yaw_angle_rad = np.radians(yaw_angle)

    # 创建旋转矩阵
    rotation_matrix_x = np.array([[1, 0, 0],
                                  [0, np.cos(roll_angle_rad), -np.sin(roll_angle_rad)],
                                  [0, np.sin(roll_angle_rad), np.cos(roll_angle_rad)]])

    rotation_matrix_y = np.array([[np.cos(pitch_angle_rad), 0, np.sin(pitch_angle_rad)],
                                  [0, 1, 0],
                                  [-np.sin(pitch_angle_rad), 0, np.cos(pitch_angle_rad)]])

    rotation_matrix_z = np.array([[np.cos(yaw_angle_rad), -np.sin(yaw_angle_rad), 0],
                                  [np.sin(yaw_angle_rad), np.cos(yaw_angle_rad), 0],
                                  [0, 0, 1]])

    # 组合三个旋转矩阵
    rotation_matrix = np.dot(rotation_matrix_z, np.dot(rotation_matrix_y, rotation_matrix_x))

    # 使用仿射变换旋转图像
    rotated_image = cv2.warpAffine(rgb_image, rotation_matrix[:2, :], (rgb_image.shape[1], rgb_image.shape[0]))
    return rotated_image

def transform_perspective(img,label_image):

    # 定义输入图像的四个顶点坐标
    src_pts = np.float32([[0, 0], [img.shape[1], 0], [img.shape[1], img.shape[0]], [0, img.shape[0]]])

    # 定义输出图像的四个顶点坐标，可以根据需要进行调整
    # dst_pts = np.float32([[10, 10], [img.shape[1] - 10, 10], [img.shape[1] - 100, img.shape[0] - 100], [100, img.shape[0] - 100]])
    dst_pts = np.float32(
        [[100, 100], [img.shape[1] - 100, 100], [img.shape[1] - 10, img.shape[0] - 10], [10, img.shape[0] - 10]])
    # 计算透视变换矩阵
    perspective_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # 进行透视变换
    result1 = cv2.warpPerspective(img, perspective_matrix, (img.shape[1], img.shape[0]))
    result2 = cv2.warpPerspective(label_image, perspective_matrix, (label_image.shape[1], label_image.shape[0]))
    # 保存输出图像
    return result1, result2

def addguise(rotated_image):
    # 添加高斯噪声
    noise = np.random.normal(0, 0.1, rotated_image.shape).astype(np.uint8)
    noisy_image = cv2.add(rotated_image, noise)
    return noisy_image

def img_totensor(rgb_image_ori):
    rgb_image_ori = rgb_image_ori.astype(np.float32) / 255
    rgb_image_ori = transforms.ToTensor()(rgb_image_ori)
    rgb_image_ori = squezzimg(rgb_image_ori)
    return rgb_image_ori

def readtxt():
    file_path = "dataset/pothole_label.txt"  # 替换为你的文本文件路径

    # 打开文本文件
    with open(file_path, 'r') as file:
        # 读取文件的所有行
        lines = file.readlines()

    # 显示每一行
    idxy = []
    name_ls = []
    for line in lines:
        code = line.strip().split('|')
        if '|' not in line:
            name_ls.append(code[0][:-1])
            idxy.append([])
            continue
        name_ls.append(code[0])
        idxy.append(code[1:-1])

    return name_ls, idxy


def squezzimg(rgb_image):
    if rgb_image.size()[1] > 3000 or rgb_image.size()[2] > 3000:
        # 使用 interpolate 函数进行缩放
        target_size = (rgb_image.size()[1] // 2, rgb_image.size()[2] // 2)
        output_tensor = F.interpolate(rgb_image.unsqueeze(0), target_size, mode='bilinear',
                                      align_corners=False)
        rgb_image = output_tensor.squeeze(0)
        return rgb_image
    return rgb_image


class potholedataset(BaseDataset):
    """dataloader for pothole dataset"""

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.batch_size = opt.batch_size
        self.num_labels = 2

        if opt.phase == "train":
            self.image_list = np.arange(1, 501)
            random.shuffle(self.image_list)
        elif opt.phase == "test" or opt.phase == "val":
            self.image_list = np.arange(501, 601)
        else:
            self.image_list, self.mask_idxy = readtxt()

    def __getitem__(self, index):
        base_dir = "./dataset/pothole/"

        if self.opt.phase == "ours":
            name = self.image_list[index]
            idxy_ls = self.mask_idxy[index]
            rgb_image_ori = cv2.cvtColor(cv2.imread(base_dir+name), cv2.COLOR_BGR2RGB)

            if idxy_ls != []:
                rgb_image = rgb_image_ori
                rgb_image_ori = img_totensor(rgb_image_ori)
                # 创建与输入图像大小相同的黑色图像
                h, w, _ = rgb_image.shape
                black_image = np.zeros_like(rgb_image)
                for st1 in idxy_ls:
                    x1, y1, x2, y2 = map(int, st1.split(','))
                    # 在黑色图像上绘制矩形
                    cv2.rectangle(black_image, (x1, y1), (x2, y2), (255, 255, 255), thickness=-1)

                # 提取矩形对应的颜色
                rgb_image = cv2.bitwise_and(rgb_image, black_image)
                rgb_image = img_totensor(rgb_image)
                return {'rgb_image': rgb_image, 'path': name, 'ori_image': rgb_image_ori}

            rgb_image = img_totensor(rgb_image_ori)
            return {'rgb_image': rgb_image, 'path': name, 'ori_image': []}
        else:
            name = str(self.image_list[index]).zfill(4) + ".png"
            rgb_image = cv2.cvtColor(cv2.imread(os.path.join(base_dir, 'rgb1', name)), cv2.COLOR_BGR2RGB)
            label_image = cv2.cvtColor(cv2.imread(os.path.join(base_dir, 'label1', name)), cv2.COLOR_BGR2RGB)

            if np.random.rand() < 0.2:
                # 旋转
                anglex = np.random.uniform(0, 30)
                angley = np.random.uniform(0, 30)
                anglez = np.random.uniform(0, 30)
                rgb_image, label_image = rotate(rgb_image, anglex, angley, anglez), rotate(label_image, anglex, angley,
                                                                                           anglez)
                # 显示原始图像和欧拉角旋转后的图像
            if np.random.rand() < 0.2:
                rgb_image, label_image = transform_perspective(rgb_image, label_image)
            if np.random.rand() < 0.2:
                # 添加高斯噪声
                rgb_image = addguise(rgb_image)
            if np.random.rand() < 0.2:
                # 添加高斯模糊
                rgb_image = cv2.GaussianBlur(rgb_image, (29, 29), 0)

            rgb_image = rgb_image.astype(np.float32) / 255
            rgb_image = transforms.ToTensor()(rgb_image)

            label = np.zeros((label_image.shape[0], label_image.shape[1]), dtype=np.uint8)
            label[label_image[:, :, 0] > 0] = 1
            label = torch.from_numpy(label)
            label = label.type(torch.LongTensor)
            return {'rgb_image': rgb_image, 'label': label,
                    'path': name}

    def __len__(self):
        return len(self.image_list)

    def name(self):
        return 'pothole'
