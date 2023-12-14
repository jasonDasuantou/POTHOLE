import albumentations as A
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import os
import torch.utils.data as data
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from .data.transforms import DenoiseTransform

# 默认输入网络的图片大小
IMAGE_SIZE = 200

# 定义一个转换关系，用于将图像数据转换成PyTorch的Tensor形式

dataTransform = transforms.Compose([
    transforms.Resize([IMAGE_SIZE, IMAGE_SIZE]),  # 将图像按比例缩放至合适尺寸
    # transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),        # 从图像中心裁剪合适大小的图像
    transforms.ToTensor(),  # 转换成Tensor形式，并且数值归一化到[0.0, 1.0]，同时将H×W×C的数据转置成C×H×W，这一点很关键
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

dataTransform1 = transforms.Compose([
    # transforms.RandomCrop(IMAGE_SIZE),                         # 将图像按比例缩放至合适尺寸       # 从图像中心裁剪合适大小的图像
    transforms.ToTensor(),  # 转换成Tensor形式，并且数值归一化到[0.0, 1.0]，同时将H×W×C的数据转置成C×H×W，这一点很关键
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

dataTransform2 = A.Compose(
    [
        A.HorizontalFlip(p = 0.5),
        # A.Transpose(),
        A.OneOf([
            A.GaussNoise(),
        ], p=0.3),
        A.OneOf([
            A.MotionBlur(p=0.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=15, p=1),
        ], p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=0.1),
            A.PiecewiseAffine(p=0.3),
        ], p=0.5),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.Sharpen(),
            A.Emboss(),
            A.RandomBrightnessContrast(),
        ], p=0.5),
        A.HueSaturationValue(p=0.3),
    ])

def generate_gaussian_noise(size,  mean=0.1, sigma=0.1):
    return np.random.normal(mean, sigma, size).astype(dtype=np.float32)

class DogsVSCatsDataset(data.Dataset):  # 新建一个数据集类，并且需要继承PyTorch中的data.Dataset父类
    def __init__(self, mode, dir):  # 默认构造函数，传入数据集类别（训练或测试），以及数据集路径
        self.mode = mode
        self.list_img = []  # 新建一个image list，用于存放图片路径，注意是图片路径
        self.list_label = []  # 新建一个label list，用于存放图片对应猫或狗的标签，其中数值0表示猫，1表示狗
        self.data_size = 0  # 记录数据集大小
        self.transform1 = DenoiseTransform(augment=True)  # 转换关系
        self.transform2 = DenoiseTransform(patch_size=200)


        if self.mode == 'train':  # 训练集模式下，需要提取图片的路径和标签
            dir = dir + '/train1/'  # 训练集路径在"dir"/train/
            for file in os.listdir(dir):  # 遍历dir文件夹
                self.list_img.append(dir + file)  # 将图片路径和文件名添加至image list
                self.data_size += 1  # 数据集增1
                name = file.split(sep='.')  # 分割文件名，"cat.0.jpg"将分割成"cat",".","jpg"3个元素
                # label采用one-hot编码，"1,0"表示猫，"0,1"表示狗，任何情况只有一个位置为"1"，在采用CrossEntropyLoss()计算Loss情况下，label只需要输入"1"的索引，即猫应输入0，狗应输入1
                if 'normal' in name[0]:
                    self.list_label.append(1)  # 图片为正常路面，label为1
                else:
                    self.list_label.append(0)  # 图片为坑洼路面，label为0，注意：list_img和list_label中的内容是一一配对的
        elif self.mode == 'test':  # 测试集模式下，只需要提取图片路径就行
            dir = dir   # 测试集路径为"dir"/test/
            for file in os.listdir(dir):
                self.list_img.append(dir + file)  # 添加图片路径至image list
                self.data_size += 1
                name = file.split(sep='.')
                if 'normal' in name[0]:
                    self.list_label.append(1)  # 图片为正常路面，label为1
                else:
                    self.list_label.append(0)  # 图片为坑洼路面，label为0
        else:
            print('Undefined Dataset!')

    def __getitem__(self,
                    item):  # 重载data.Dataset父类方法，获取数据集中数据内容                                     # 训练集模式下需要读取数据集的image和label
        item = item % self.data_size
        img = Image.open(self.list_img[item])
        height = img.height
        width = img.width
        label = self.list_label[item]
        if self.mode =='test':
            return dataTransform(img), torch.LongTensor([label]), self.list_img[item]
        # if self.mode =='test':
        #    return dataTransform1(img), torch.LongTensor([label])
        if label == 1 and height > 200 and width > 200:
            img = np.array(img)
            img = self.transform2(img) #随机裁剪
            img = Image.fromarray(np.uint8(img))
            return dataTransform1(img), torch.LongTensor([label])
        img_arr = np.array(img)
        img = dataTransform2(image = img_arr)['image']
        img = Image.fromarray(np.uint8(img))

        tensor_img =dataTransform(img)
        # tensor_img = tensor_img + generate_gaussian_noise(tensor_img.size(), mean=0, sigma=0.03)

        return tensor_img, torch.LongTensor([label])

    def __len__(self):
        if self.mode == 'train':
            return self.data_size * 1  # 返回数据集大小
        else:
            return self.data_size
