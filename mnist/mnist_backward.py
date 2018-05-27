# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import mnist_forward

import os
import time
import numpy as np
#from visdom import Visdom

# 参数设置
LR = 0.01 
MOMENTUM = 0.5
BATCH_SIZE = 64
EPOCHS = 10
start_epoch = 1
MODEL_SAVE_PATH="./model/mnist_model.pt"   # 模型保存路径

device = torch.device("cpu")

# 训练数据预处理
train_transform = transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,)),
                   ])

train_dataset = datasets.MNIST('./data', train=True, download=False,transform = train_transform)

# 载入训练数据
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=BATCH_SIZE, shuffle=True,)

train_data = torch.unsqueeze(train_dataset.train_data,1)[:1500]
train_label = train_dataset.train_labels[:1500]
# 可视化
#viz = Visdom()


#viz.image(train_data[:100])
#time.sleep(1)
#line = viz.line(np.arange(10))
#time.sleep(.2)

# 模型实例化
model = mnist_forward.Net()

# 定义优化器
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)

#start_time = time.time()
#time_p,loss_p,tr_acc = [],[],[],

# 数据窗口
#text = viz.text("<h1>convolution Nueral Network</h1>")

# 断点续训
if os.path.isfile(MODEL_SAVE_PATH):
    print("=> loading checkpoint '{}'".format(MODEL_SAVE_PATH))  
    checkpoint = torch.load(MODEL_SAVE_PATH)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


def train(epoch):
    # 模型切换到训练模式
    model.train()
    
    sum_loss,sum_acc,sum_step = 0.,0.,0.
    for i, (data, target) in enumerate(train_loader):

        # 梯度清零
        optimizer.zero_grad()
        # 前向传播
        output = model.forward(data)
        # 计算损失函数
        loss = F.nll_loss(output, target)
        '''
        sum_loss += loss.data[0]*len(target)
        pred_tr = torch.max(output,1)[1]
        sum_acc +=sum(pred_tr==target).data[0].item()
        sum_step += target.size(0)

        time_p.append(time.time()-start_time)
        tr_acc.append(sum_acc/sum_step)
        loss_p.append(sum_loss/sum_step)
        '''

        # 反向传播
        loss.backward()
        # 参数更新
        optimizer.step()
        ''' 
        viz.line(X =np.column_stack((np.array(time_p),np.array(time_p))),
                Y=np.column_stack((np.array(loss_p),np.array(tr_acc))),
                win=line,
                opts=dict(legend=['loss','tr_acc']))
        '''
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, i * len(data), len(train_loader.dataset),
            100. * i / len(train_loader), loss.item()))
        '''
        viz.text("<p style='color:red'>epoch:{}</p><br><p style='color:blue'>Loss:{:.4f}</p><br>"
                "<p style='color:BlueViolet'>tr_acc:{:.4f}</p>"
                "<p style='color:green'>Time:{:.2f}</p>".format(epoch, sum_loss/sum_step, sum_acc/sum_step,time.time()-start_time),win=text)
        sum_loss,sum_acc,sum_step = 0.,0.,0.,
        '''
        # 保存模型
        torch.save({'epoch':epoch+1,
            'state_dict':model.state_dict(),
            'optimizer':optimizer.state_dict(),
            },MODEL_SAVE_PATH)

        
# 启动程序
if __name__ == '__main__':
    for epoch in range(start_epoch,EPOCHS+1):
        train(epoch)
        
