#import sys
import os
import time
import math
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
#import itertools
import struct # get_image_size
import imghdr # get_image_size
import random
from torch.utils.data import Dataset

# 数据导入预处理类
class listDataset(Dataset):
    def __init__(self, root, shape=None, shuffle=True, transform=None, target_transform=None, train=False, seen=0, batch_size=64, num_workers=4):
       with open(root, 'r') as file:
           self.lines = file.readlines()

       if shuffle:
           random.shuffle(self.lines)

       self.nSamples  = len(self.lines)
       self.transform = transform
       self.target_transform = target_transform
       self.train = train
       self.shape = shape
       self.seen = seen
       self.batch_size = batch_size
       self.num_workers = num_workers

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        imgpath = self.lines[index].rstrip()
        print('imgpath:@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',imgpath)

        if self.train and index % 64== 0:
            print('index',index)
            if self.seen < 4000*64:
               width = 13*32
               self.shape = (width, width)
            elif self.seen < 8000*64:
               width = (random.randint(0,3) + 13)*32
               self.shape = (width, width)
            elif self.seen < 12000*64:
               width = (random.randint(0,5) + 12)*32
               self.shape = (width, width)
            elif self.seen < 16000*64:
               width = (random.randint(0,7) + 11)*32
               self.shape = (width, width)
            else: # self.seen < 20000*64:
               width = (random.randint(0,9) + 10)*32
               self.shape = (width, width)
        if self.train:
            # 训练
            jitter = 0.2
            hue = 0.1
            saturation = 1.5 
            exposure = 1.5

            img, label = load_data_detection(imgpath, self.shape, jitter, hue, saturation, exposure)
            label = torch.from_numpy(label)
        else:
            # 测试
            img = Image.open(imgpath).convert('RGB')
            if self.shape:
                img = img.resize(self.shape)
    
            labpath = imgpath.replace('images', 'labels').replace('JPEGImages', 'labels').replace('.jpg', '.txt').replace('.png','.txt')
            label = torch.zeros(50*5)
            try:

                tmp = torch.from_numpy(read_truths_args(labpath, 8.0/img.width).astype('float32'))
            except Exception:
                tmp = torch.zeros(1,5)
            tmp = tmp.view(-1)
            tsz = tmp.numel()
            if tsz > 50*5:
                label = tmp[0:50*5]
            elif tsz > 0:
                label[0:tsz] = tmp

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        self.seen = self.seen + self.num_workers
        print('listDataset img',img.shape)
        print('listDataset label',label.shape)
        return (img, label)
# 图像处理
def distort_image(im, hue, sat, val):
    print('distort_image input im',im)
    print('distort_image input hue',hue)
    print('distort_image input sat',sat)
    print('distort_image input val',val)
    im = im.convert('HSV')
    cs = list(im.split())
    cs[1] = cs[1].point(lambda i: i * sat)
    cs[2] = cs[2].point(lambda i: i * val) 
    def change_hue(x):
        x += hue*255
        if x > 255:
            x -= 255
        elif x < 0:
            x += 255
        return x
    cs[0] = cs[0].point(change_hue)
    im = Image.merge(im.mode, tuple(cs))
    im = im.convert('RGB')
    print('distort_image output im',im)
    return im
# 图像处理
def rand_scale(s):
    print('rand_scale input s',s)
    scale = random.uniform(1, s)
    if(random.randint(1,10000)%2): 
        return scale
    print('rand_scale output s',1./scale)
    return 1./scale
# 图像处理
def random_distort_image(im, hue, saturation, exposure):
    dhue = random.uniform(-hue, hue)
    dsat = rand_scale(saturation)
    dexp = rand_scale(exposure)
    res = distort_image(im, dhue, dsat, dexp)
    return res

# 图像缩放处理
def data_augmentation(img, shape, jitter, hue, saturation, exposure):
    print('data_augmentation input img, shape, jitter, hue, saturation, exposure',img, shape, jitter, hue, saturation, exposure)
    oh = img.height  
    ow = img.width
    
    dw =int(ow*jitter)
    dh =int(oh*jitter)

    pleft  = random.randint(-dw, dw)
    pright = random.randint(-dw, dw)
    ptop   = random.randint(-dh, dh)
    pbot   = random.randint(-dh, dh)

    swidth =  ow - pleft - pright
    sheight = oh - ptop - pbot

    sx = float(swidth)  / ow
    sy = float(sheight) / oh
    
    flip = random.randint(1,10000)%2
    cropped = img.crop( (pleft, ptop, pleft + swidth - 1, ptop + sheight - 1))

    dx = (float(pleft)/ow)/sx
    dy = (float(ptop) /oh)/sy

    sized = cropped.resize(shape)

    if flip: 
        sized = sized.transpose(Image.FLIP_LEFT_RIGHT)
    img = random_distort_image(sized, hue, saturation, exposure)
    print('data_augmentation outputimg, flip, dx,dy,sx,sy ',img, flip, dx,dy,sx,sy)
    return img, flip, dx,dy,sx,sy 
# 操作训练数据:处理标签数据
def fill_truth_detection(labpath, w, h, flip, dx, dy, sx, sy):
    print('fill_truth_detection input labpath, w, h, flip, dx, dy, sx, sy',labpath, w, h, flip, dx, dy, sx, sy)
    max_boxes = 50
    label = np.zeros((max_boxes,5))
    if os.path.getsize(labpath):
        bs = np.loadtxt(labpath)
        if bs is None:
            return label
        bs = np.reshape(bs, (-1, 5))
        cc = 0
        for i in range(bs.shape[0]):
            x1 = bs[i][1] - bs[i][3]/2
            y1 = bs[i][2] - bs[i][4]/2
            x2 = bs[i][1] + bs[i][3]/2
            y2 = bs[i][2] + bs[i][4]/2
            
            x1 = min(0.999, max(0, x1 * sx - dx)) 
            y1 = min(0.999, max(0, y1 * sy - dy)) 
            x2 = min(0.999, max(0, x2 * sx - dx))
            y2 = min(0.999, max(0, y2 * sy - dy))
            
            bs[i][1] = (x1 + x2)/2
            bs[i][2] = (y1 + y2)/2
            bs[i][3] = (x2 - x1)
            bs[i][4] = (y2 - y1)

            if flip:
                bs[i][1] =  0.999 - bs[i][1] 
            
            if bs[i][3] < 0.001 or bs[i][4] < 0.001:
                continue
            label[cc] = bs[i]
            cc += 1
            if cc >= 50:
                break

    label = np.reshape(label, (-1))
    print('fill_truth_detection label output',np.array(label).shape)
    return label

# 训练:操作训练数据
def load_data_detection(imgpath, shape, jitter, hue, saturation, exposure):
    print('load_data_detection input imgpath',imgpath, shape, jitter, hue, saturation, exposure)
    print('load_data_detection input shape, jitter, hue, saturation, exposure',shape, jitter, hue, saturation, exposure)
    labpath = imgpath.replace('images', 'labels').replace('JPEGImages', 'labels').replace('.jpg', '.txt').replace('.png','.txt')

    ## data augmentation
    print('imgpath:=======================================',imgpath)
    img = Image.open(imgpath).convert('RGB')
    img,flip,dx,dy,sx,sy = data_augmentation(img, shape, jitter, hue, saturation, exposure)
    label = fill_truth_detection(labpath, img.width, img.height, flip, dx, dy, 1./sx, 1./sy)
    print('labpath:+++++++++++++++++++++++++++++++++++++++',labpath)
    print('load_data_detection output',img)
    print('load_data_detection output label',label.shape)
    return img,label

# 用F.sigmod()代替
def sigmoid(x):
    return 1.0/(math.exp(-x)+1.)


# 计算交并比
def bbox_iou(box1, box2, x1y1x2y2=True):
#    print('yolo bbox_iou input box1',box1)
#    print('yolo bbox_iou input box2',box2)
    if x1y1x2y2:
        x1_min = min(box1[0], box2[0])
        x2_max = max(box1[2], box2[2])
        y1_min = min(box1[1], box2[1])
        y2_max = max(box1[3], box2[3])
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    else:
        w1, h1 = box1[2], box1[3]
        w2, h2 = box2[2], box2[3]
        x1_min = min(box1[0]-w1/2.0, box2[0]-w2/2.0)
        x2_max = max(box1[0]+w1/2.0, box2[0]+w2/2.0)
        y1_min = min(box1[1]-h1/2.0, box2[1]-h2/2.0)
        y2_max = max(box1[1]+h1/2.0, box2[1]+h2/2.0)

    w_union = x2_max - x1_min
    h_union = y2_max - y1_min
    w_cross = w1 + w2 - w_union
    h_cross = h1 + h2 - h_union
    carea = 0
    if w_cross <= 0 or h_cross <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    carea = w_cross * h_cross
    uarea = area1 + area2 - carea
    print('yolo bbox_iou output carea/uarea',float(carea/uarea))
    return float(carea/uarea)

# 多个boxes计算交并比
def multi_bbox_ious(boxes1, boxes2, x1y1x2y2=True):
#    print('yolo multi_bbox_iou input boxes1',boxes1.shape)
#    print('yolo multi_bbox_iou input boxes2',boxes2.shape)
    if x1y1x2y2:
        x1_min = torch.min(boxes1[0], boxes2[0])
        x2_max = torch.max(boxes1[2], boxes2[2])
        y1_min = torch.min(boxes1[1], boxes2[1])
        y2_max = torch.max(boxes1[3], boxes2[3])
        w1, h1 = boxes1[2] - boxes1[0], boxes1[3] - boxes1[1]
        w2, h2 = boxes2[2] - boxes2[0], boxes2[3] - boxes2[1]
    else:
        w1, h1 = boxes1[2], boxes1[3]
        w2, h2 = boxes2[2], boxes2[3]
        x1_min = torch.min(boxes1[0]-w1/2.0, boxes2[0]-w2/2.0)
        x2_max = torch.max(boxes1[0]+w1/2.0, boxes2[0]+w2/2.0)
        y1_min = torch.min(boxes1[1]-h1/2.0, boxes2[1]-h2/2.0)
        y2_max = torch.max(boxes1[1]+h1/2.0, boxes2[1]+h2/2.0)

    w_union = x2_max - x1_min
    h_union = y2_max - y1_min
    w_cross = w1 + w2 - w_union
    h_cross = h1 + h2 - h_union
    mask = (((w_cross <= 0) + (h_cross <= 0)) > 0)
    area1 = w1 * h1
    area2 = w2 * h2
    carea = w_cross * h_cross
    carea[mask] = 0
    uarea = area1 + area2 - carea
#    print('yolo multi_bbox_ious output carea/uarea',(carea/uarea).shape)
    return carea/uarea

# 极大值抑制算法，只用在detect阶段
def nms(boxes, nms_thresh):
    print('nms input boxes',np.array(boxes).shape)
    print('nms input nms_thresh',nms_thresh)
    if len(boxes) == 0:
        return boxes

    det_confs = torch.zeros(len(boxes))
    print('det_confs====*********========',det_confs.shape)
    for i in range(len(boxes)):
        print('i++++++++++++++++++++',i)
        # 取出 1 - 目标分数 给det_confs 并转换为列（自然转换） 
        det_confs[i] = 1-boxes[i][4]                
    
    # 从小到大排序（因为前面 -1,所以取最小值）
    _,sortIds = torch.sort(det_confs)
    out_boxes = []
    for i in range(len(boxes)):
        print('i====================',i)
        # 找出目标分数最小的box
        box_i = boxes[sortIds[i]]
        if box_i[4] > 0:
            # 如果目标分数大于0，则保存
            out_boxes.append(box_i)
            # 依次取出其他 10647-1 个box 并与 目标分数最大的box 求交并比
            for j in range(i+1, len(boxes)):
                print('range(i+1, len(boxes))',range(i+1, len(boxes)))
                box_j = boxes[sortIds[j]]
                # 如果交并比 > 0.4 ==nms 阈值，则表示预测的是同一个object，box 置零(目标分数置零)
#                print('confidence is > nms_thresh',bbox_iou(box_i, box_j, x1y1x2y2=False) > nms_thresh)
                if bbox_iou(box_i, box_j, x1y1x2y2=False) > nms_thresh:
                    #print(box_i, box_j, bbox_iou(box_i, box_j, x1y1x2y2=False))
                    box_j[4] = 0
    print('nms output out_boxes',np.array(out_boxes).shape)
    return out_boxes
# 训练 train阶段
def convert2cpu(gpu_matrix):
#    print('convert2cpu')
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)
# 训练 test阶段
def convert2cpu_long(gpu_matrix):
#    print('convert2cpu_long')
    return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)

def get_all_boxes(output, conf_thresh, num_classes, only_objectness=1, validation=False):
    # total number of inputs (batch size)
    # first element (x) for first tuple (x, anchor_mask, num_anchor)
    print('get_all_boxes input output',output.keys())
    print('get_all_boxes input num_classes',num_classes)
    
    tot = output[0]['x'].data.size(0)
    all_boxes = [[] for i in range(tot)]
    for i in range(len(output)):
        pred, anchors, num_anchors = output[i]['x'].data, output[i]['a'], output[i]['n'].item()
        print('get_all_boxes pred',pred)
        b = get_region_boxes(pred, conf_thresh, num_classes, anchors, num_anchors, \
                only_objectness=only_objectness, validation=validation)
        print('b',b)
        for t in range(tot):
            all_boxes[t] += b[t]
    print('get_all_boxes output all_boxes',all_boxes)
    return all_boxes
# 训练 测试环节
def get_region_boxes(output, conf_thresh, num_classes, anchors, num_anchors, only_objectness=1, validation=False):
    print('get_region_boxes input output',output.shape)
    print('get_region_boxes input conf_thresh',conf_thresh)
    print('get_region_boxes input num_classes',num_classes)
    print('get_region_boxes input anchors',anchors)
    print('get_region_boxes input num_anchors',num_anchors)
    device = torch.device("cpu")
    anchors = anchors.to(device)
    anchor_step = anchors.size(0)//num_anchors
    if output.dim() == 3:
        output = output.unsqueeze(0)
    batch = output.size(0)
    assert(output.size(1) == (5+num_classes)*num_anchors)
    h = output.size(2)
    w = output.size(3)
    cls_anchor_dim = batch*num_anchors*h*w

    all_boxes = []
    output = output.view(batch*num_anchors, 5+num_classes, h*w).transpose(0,1).contiguous().view(5+num_classes, cls_anchor_dim)

    grid_x = torch.linspace(0, w-1, w).repeat(batch*num_anchors, h, 1).view(cls_anchor_dim).to(device)
    grid_y = torch.linspace(0, h-1, h).repeat(w,1).t().repeat(batch*num_anchors, 1, 1).view(cls_anchor_dim).to(device)
    ix = torch.LongTensor(range(0,2)).to(device)
    anchor_w = anchors.view(num_anchors, anchor_step).index_select(1, ix[0]).repeat(1, batch, h*w).view(cls_anchor_dim)
    anchor_h = anchors.view(num_anchors, anchor_step).index_select(1, ix[1]).repeat(1, batch, h*w).view(cls_anchor_dim)

    xs, ys = torch.sigmoid(output[0]) + grid_x, torch.sigmoid(output[1]) + grid_y
    ws, hs = torch.exp(output[2]) * anchor_w.detach(), torch.exp(output[3]) * anchor_h.detach()
    det_confs = torch.sigmoid(output[4])

    # by ysyun, dim=1 means input is 2D or even dimension else dim=0
    cls_confs = torch.nn.Softmax(dim=1)(output[5:5+num_classes].transpose(0,1)).detach()
    cls_max_confs, cls_max_ids = torch.max(cls_confs, 1)
    cls_max_confs = cls_max_confs.view(-1)
    cls_max_ids = cls_max_ids.view(-1)
    
    sz_hw = h*w
    sz_hwa = sz_hw*num_anchors
    det_confs = convert2cpu(det_confs)
    cls_max_confs = convert2cpu(cls_max_confs)
    cls_max_ids = convert2cpu_long(cls_max_ids)
    xs, ys = convert2cpu(xs), convert2cpu(ys)
    ws, hs = convert2cpu(ws), convert2cpu(hs)
    if validation:
        cls_confs = convert2cpu(cls_confs.view(-1, num_classes))

    for b in range(batch):
        boxes = []
        for cy in range(h):
            for cx in range(w):
                for i in range(num_anchors):
                    ind = b*sz_hwa + i*sz_hw + cy*w + cx
                    det_conf =  det_confs[ind]
                    if only_objectness:
                        conf = det_confs[ind]
                    else:
                        conf = det_confs[ind] * cls_max_confs[ind]
    
                    if conf > conf_thresh:
                        bcx = xs[ind]
                        bcy = ys[ind]
                        bw = ws[ind]
                        bh = hs[ind]
                        cls_max_conf = cls_max_confs[ind]
                        cls_max_id = cls_max_ids[ind]
                        box = [bcx/w, bcy/h, bw/w, bh/h, det_conf, cls_max_conf, cls_max_id]
                        if (not only_objectness) and validation:
                            for c in range(num_classes):
                                tmp_conf = cls_confs[ind][c]
                                if c != cls_max_id and det_confs[ind]*tmp_conf > conf_thresh:
                                    box.append(tmp_conf)
                                    box.append(c)
                        boxes.append(box)
        all_boxes.append(boxes)
    print('get_region_boxes output all_boxes',np.array(all_boxes).shape)
    return all_boxes

# 绘制边框
def plot_boxes(img, boxes, savename=None, class_names=None):
    print('plot_boxes input img',img)
    print('plot_boxes input boxes',np.array(boxes).shape)
    colors = torch.FloatTensor([[1,0,1],[0,0,1],[0,1,1],[0,1,0],[1,1,0],[1,0,0]])
    def get_color(c, x, max_val):
        ratio = float(x)/max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1-ratio) * colors[i][c] + ratio*colors[j][c]
        return int(r*255)

    width = img.width
    height = img.height
    draw = ImageDraw.Draw(img)
    
    print("%d box(es) is(are) found" % len(boxes))
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = (box[0] - box[2]/2.0) * width
        y1 = (box[1] - box[3]/2.0) * height
        x2 = (box[0] + box[2]/2.0) * width
        y2 = (box[1] + box[3]/2.0) * height

        rgb = (255, 0, 0)
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            print('%s: %f' % (class_names[cls_id], cls_conf))
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red   = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue  = get_color(0, offset, classes)
            rgb = (red, green, blue)
            draw.text((x1, y1), class_names[cls_id], fill=rgb)
        draw.rectangle([x1, y1, x2, y2], outline=rgb)
    if savename:
        print("save plot results to %s" % savename)
        img.save(savename)
    print('plot_boxes output',img)
    return img

def read_truths(lab_path):
    print('read_truths',lab_path)
    if not os.path.exists(lab_path):
        return np.array([])
    if os.path.getsize(lab_path):
        truths = np.loadtxt(lab_path)
        truths = truths.reshape(truths.size//5, 5) # to avoid single truth problem
        return truths
    else:
        return np.array([])

def read_truths_args(lab_path, min_box_scale):
    truths = read_truths(lab_path)
    new_truths = []
    for i in range(truths.shape[0]):
        if truths[i][3] < min_box_scale:
            continue
        new_truths.append([truths[i][0], truths[i][1], truths[i][2], truths[i][3], truths[i][4]])
    return np.array(new_truths)

# 加载类别名称(detect阶段)
def load_class_names(namesfile):
    class_names = []
    with open(namesfile, 'r', encoding='utf8') as fp:
        lines = fp.readlines()
    for line in lines:
        class_names.append(line.strip())
    print('load_class_names output class_names',class_names)
    return class_names
# 图像转tensor
def image2torch(img):
    if isinstance(img, Image.Image):
        width = img.width
        height = img.height
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
        img = img.view(height, width, 3).transpose(0,1).transpose(0,2).contiguous()
        img = img.view(1, 3, height, width)
        img = img.float().div(255.0)
    elif type(img) == np.ndarray: # cv2 image
        img = torch.from_numpy(img.transpose(2,0,1)).float().div(255.0).unsqueeze(0)
    else:
        print("unknown image type")
        exit(-1)
    return img

#import types
def do_detect(model, img, conf_thresh, nms_thresh):
#    print('do_detect input model------------------------',model)
    print('do_detect input img',img)

    model.eval()
    img = image2torch(img)
    img = img.to(torch.device("cpu"))
    out_boxes = model.forward(img)
#    print('do_detect out_boxes',out_boxes)
    boxes = get_all_boxes(out_boxes, conf_thresh, model.num_classes)[0]
    print('do_detect get_all_boxes boxes',boxes)
    boxes = nms(boxes, nms_thresh)
    print('do_detect output boxes',boxes)
    return boxes

# 转移到cfg.py
# 解析 cfg/voc.data 文件
def read_data_cfg(datacfg):
    print('read_data_cfg input:datacfg !!!!',datacfg)
    options = dict()
#    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '1'
    with open(datacfg, 'r') as fp:
        lines = fp.readlines()

    for line in lines:
        line = line.strip()
        if line == '':
            continue
        key,value = line.split('=')
        key = key.strip()
        value = value.strip()
        options[key] = value
    print('read_data_cfg output:options',options)
    return options
#
#def scale_bboxes(bboxes, width, height):
#    import copy
#    dets = copy.deepcopy(bboxes)
#    for i in range(len(dets)):
#        dets[i][0] = dets[i][0] * width
#        dets[i][1] = dets[i][1] * height
#        dets[i][2] = dets[i][2] * width
#        dets[i][3] = dets[i][3] * height
#    return dets


