import shutil
import torch
import torchvision
from tqdm import tqdm
from src.getdata import DogsVSCatsDataset as DVCD
from torch.utils.data import DataLoader as DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import csv

def create_model():
    mode1_vgg16 = torchvision.models.vgg16(pretrained=True)
    # 读取输入特征的维度（因为这一维度不需要修改）
    num_fc = mode1_vgg16.classifier[6].in_features
    # 修改最后一层的输出维度，即分类数
    mode1_vgg16.classifier[6] = torch.nn.Linear(num_fc, 2)

    # 读取输入特征的维度（因为这一维度不需要修改）
    num_fc = mode1_vgg16.classifier[6].in_features
    # 修改最后一层的输出维度，即分类数
    mode1_vgg16.classifier[6] = torch.nn.Linear(num_fc, 2)

    # 对于模型的每个权重，使其不进行反向传播，即固定参数
    for param in mode1_vgg16.parameters():
        param.requires_grad = False
    # 但是参数全部固定了，也没法进行学习，所以我们不固定最后一层，即全连接层
    for param in mode1_vgg16.classifier[6].parameters():
        param.requires_grad = True
    return mode1_vgg16

def test(batch_size):
    acc = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    # setting model
    model = create_model()  # 实例化一个网络
    model.to(device)  # 送入GPU，利用GPU计算
    model.load_state_dict(torch.load(model_file))  # 加载训练好的模型参数
    model.eval()  # 设定为评估模式，即计算过程中不要dropout

    # get data
    datafile_val = DVCD('test', dataset_dir)
    num = len(datafile_val)
    num_test = 0
    print('Dataset loaded! length of test set is {0}'.format(num))
    dataloader_val = DataLoader(datafile_val, batch_size, num_workers=0)
    data_test = tqdm(dataloader_val)
    for img, label, img_index in data_test:  # 循环读取封装后的数据集，其实就是调用了数据集中的__getitem__()方法，只是返回数据格式进行了一次封装
        img, label = Variable(img).to(device), Variable(label).to(device)  # 将数据放置在PyTorch的Variable节点中，并送入GPU中作为网络计算起点
        out = model(img)  # 计算网络输出值，就是输入网络一个图像数据，输出猫和狗的概率，调用了网络中的forward()方法
        pre = F.softmax(out, dim=1)  # 清除优化器中的梯度以便下一次计算，因为优化器默认会保留，不清除的话，每次计算梯度都回累加
        ground_label_list = label.squeeze()
        pre_label_list = pre.data.cpu().numpy()

        for index in range(batch_size):
            num_test += 1
            ground_label = ground_label_list[index].data.cpu().numpy().tolist()
            pre = pre_label_list[index]
            img_name = img_index[index]

            if pre[0] > pre[1]:
                new_img_name = './dataset/pothole/'+ img_name.split('/')[-1]
            else:
                new_img_name = './dataset/none_pothole/'+ img_name.split('/')[-1]
                dic = {}
                key = img_name.split('/')[-1]
                dic[key] = '0'
                # 打开文件并进行写入
                with open('answer/ans.csv', mode='a+', newline='\n') as file:
                    # 创建 CSV 写入对象
                    writer = csv.writer(file)

                    # 写入数据
                    writer.writerows([[key, '0']])
            shutil.copy(img_name, new_img_name)

        num_ba = num_test
        data_test.set_postfix(num=num_ba)



if __name__ == '__main__':
    batch_size = 2
    dataset_dir = 'F:/数学建模922/10.28/testdata_V2/'
    model_file = './checkpoint/classification.pth'
    str1 = dataset_dir[-2]
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    acc = test(batch_size)
