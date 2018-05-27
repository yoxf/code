# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import os
import numpy as np
import mnist_forward
import mnist_backward
from PIL import Image


device = torch.device("cpu")

# 图片预处理
def pre_pic(picName):
    img = Image.open(picName) # 打开图片
    reIm = img.resize((28,28), Image.ANTIALIAS) # 裁剪模型到28*28
    im_arr = np.array(reIm.convert('L')) # 图片设置为灰度图
    threshold = 50   # 设置阈值,消除图片噪点
    for i in range(28):
        for j in range(28):
            im_arr[i][j] = 255 - im_arr[i][j] 
            if (im_arr[i][j] < threshold):  #去除噪点，小于域值的，设置为0，否则设置为255
                im_arr[i][j] = 0
            else: 
                im_arr[i][j] = 255

    nm_arr = im_arr.reshape([1, 784]) #矩阵转换为向量
    nm_arr = nm_arr.astype(np.float32) #格式转换
    img_ready = np.multiply(nm_arr, 1.0/255.0) #归一化

    return img_ready
# 加载模型
model = mnist_backward.model

# 应用
def application(data):
    model.eval()

    with torch.no_grad(): 
        output = model.forward(data)
        pred = output.max(1,keepdim=True)[1]
    return pred


# 启动程序
if __name__ == '__main__':
    path = './data/figure/'
    testNum = int(input("input the number of test pictures:"))
    for i in range(testNum):
        testPic = os.path.join(path,
                input(" the name of test picture is "))
        
        testPicArr = pre_pic(testPic)
        testPicArr = torch.from_numpy(testPicArr)
        testPicArr = testPicArr.view([-1,1,28,28])
        preValue = application(testPicArr).item()

        print("The prediction number is:",preValue)
       
