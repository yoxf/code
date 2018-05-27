# -*- coding: utf-8 -*-
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
import os

#viz = visdom.Visdom()

BATCH_SIZE = 4
LR = 0.001
EPOCHS = 10

data_transforms = {
    'train': transforms.Compose([
        # 随机切成224x224 大小图片 统一图片格式
        transforms.RandomResizedCrop(224),
        # 图像翻转
        transforms.RandomHorizontalFlip(),
        # totensor 归一化(0,255) >> (0,1)   normalize channel=（channel-mean）/std
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    "val" : transforms.Compose([
        # 图片大小缩放 统一图片格式
        transforms.Resize(256),
        # 以中心裁剪
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

data_dir = './data'
# trans data
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
# load data
data_loaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True) for x in ['train', 'val']}

# 统计文件夹下图片数
data_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
# 获取分类名称（文件夹对应名称）
class_names = image_datasets['train'].classes
#print(data_sizes, class_names)

model = models.resnet18(pretrained=True)

# 设置模型自定义参数，这里是fc层输入特征数

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)

loss_f = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

MODEL_SAVE_PATH = './model/model.pt'
train_loss, test_loss, train_acc, test_acc = [], [], [], []

# 断点续训
if os.path.isfile(MODEL_SAVE_PATH):
    print("=> loading checkpoint '{}'".format(MODEL_SAVE_PATH))  
    checkpoint = torch.load(MODEL_SAVE_PATH)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

def main():
    for epoch in range(EPOCHS):
        print(epoch)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            print(phase)
            if phase == 'train':
                scheduler.step()
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in data_loaders[phase]:
                inputs, labels = data
                # 梯度清零
                optimizer.zero_grad()
                # 前向传播
                outputs = model(inputs)
                preds = torch.max(outputs.data, 1)[1]
                loss = loss_f(outputs, labels)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data.item()*len(labels)
                running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / data_sizes[phase]
            epoch_acc = running_corrects / data_sizes[phase]

            if phase == 'val':
                test_loss.append(epoch_loss)
                test_acc.append(epoch_acc)
            else:
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)


        print("[{}/{}] train_loss:{:.3f}|train_acc:{:.3f}|test_loss:{:.3f}|test_acc{:.3f}".format(epoch+1,
        EPOCHS,train_loss[-1],train_acc[-1], test_loss[-1], test_acc[-1]))
        torch.save({'epoch':epoch+1,
            'state_dict':model.state_dict(),
            'optimizer':optimizer.state_dict(),
            },MODEL_SAVE_PATH)


if __name__ == '__main__':
    main()
