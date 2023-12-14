import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from thop import profile
from thop import clever_format
# 模型搭建  简化了的VGGnet
class vgg16(torch.nn.Module):
    def __init__(self):
        super(vgg16, self).__init__()
        self.Conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        self.Classes = torch.nn.Sequential(
            torch.nn.Linear(12 * 12 * 512, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(1024, 2))

    def forward(self, inputs):
        x = self.Conv(inputs)
        x = x.view(-1, 12 * 12 * 512)
        x = self.Classes(x)
        return x


if __name__== "__main__" :
    net=vgg16()
    x = torch.rand(1,3,200,200)
    out = net(x)
    flops, params = profile(net, inputs=(x, ))
    # print(flops, params)
    macs, params = clever_format([flops, params], "%.3f")
    print(macs,params)
    print(out.shape)
