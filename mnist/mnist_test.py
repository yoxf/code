# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import mnist_forward
import mnist_backward

TEST_BATCH_SIZE = 1000

device = torch.device("cpu")

# 测试数据预处理
test_transform = transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,)),
                   ])

# 载入测试数据
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform = test_transform),
    batch_size=TEST_BATCH_SIZE, shuffle=True,)

# 加载模型
model = mnist_backward.model

def test():
    model.eval()
    
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model.forward(data)
            test_loss += F.nll_loss(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
# 启动程序
if __name__ == '__main__':
    test()
