# -*- coding: utf-8 -*-
import os
import numpy as np
import main
from PIL import Image
from torchvision import transforms



classes = main.class_names
model = main.model

def get_names(path):
    print(path)
    names = []
    for _,_,filename in os.walk(path):
        names.append(filename)
    return names[0]

def get_image(path,names):
    images = []
    for i in names:
        img = Image.open(os.path.join(path,i))
        images.append(img)
    return images

data_transforms = transforms.Compose([
        # 图片大小缩放 统一图片格式
        transforms.Resize(256),
        # 以中心裁剪
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def img_to_tensor(images):
    tensors = []
    for i in range(len(images)):
        tensor =data_transforms(images[i])
        tensors.append(tensor)
    return tensors[0]

def application(img_loaders):
    model.train(False)
    outputs = model(img_loaders)
    preds = outputs.max(1,keepdim=True)[1].numpy().T[0]
    return preds

#if __name__ == '__main__':
#    
#    path = input(" please input image path :")
#    names = get_names(path)
#    name = input(" please input image name :")
#
#    for n in names:
#        if name == n:
#            img = get_image(path,[name])
#            img_tensor = img_to_tensor(img)
#            img_tensor = img_tensor[np.newaxis,:]
#            predValue = application(img_tensor)
#            print("the image is :", classes[predValue[0]])



