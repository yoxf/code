from __future__ import print_function

import time
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
import gc

#import dataset
from utils import *
import utils
from cfg import parse_cfg,file_lines,logging
from darknet import Darknet
import argparse

FLAGS = None
unparsed = None
device = None

# global variables
# Training settings
# Train parameters

eps           = 1e-5
keep_backup   = 5
save_interval = 1  # epoches
dot_interval  = 70  # batches

# Test parameters
evaluate = False
conf_thresh   = 0.25
nms_thresh    = 0.4
iou_thresh    = 0.5

# Training settings
def load_testlist(testlist):
    print('load_testlist',testlist)
    init_width = model.width
    init_height = model.height

    loader = torch.utils.data.DataLoader(
        utils.listDataset(testlist, shape=(init_width, init_height),
                       shuffle=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                       ]), train=False),
        batch_size=batch_size, shuffle=False)
    print('loader',loader)
    return loader

def main():
    print('main',main)
    datacfg    = FLAGS.data
    cfgfile    = FLAGS.config
    weightfile = FLAGS.weights

    data_options  = utils.read_data_cfg(datacfg)
    net_options   = parse_cfg(cfgfile)[0]
    print('net_options=============================',net_options)


    globals()["trainlist"]     = data_options['train']
    globals()["testlist"]      = data_options['valid']
    globals()["backupdir"]     = data_options['backup']
    globals()["num_workers"]   = int(data_options['num_workers'])

    globals()["batch_size"]    = int(net_options['batch'])
    globals()["max_batches"]   = int(net_options['max_batches'])
    globals()["learning_rate"] = float(net_options['learning_rate'])
    globals()["momentum"]      = float(net_options['momentum'])
    globals()["decay"]         = float(net_options['decay'])
    globals()["steps"]         = [float(step) for step in net_options['steps'].split(',')]
    globals()["scales"]        = [float(scale) for scale in net_options['scales'].split(',')]
    
    
    #Train parameters
    global max_epochs
    try:
        max_epochs = int(net_options['max_epochs'])
#        print('max_epochs----------------------------------------------------------',max_epochs)
    except KeyError:
        nsamples = file_lines(trainlist)
        
        max_epochs = (max_batches*batch_size)//nsamples+1
        print('batch_size----------------------------------------------------------',batch_size)
        print('nsamples----------------------------------------------------------',nsamples)
        print('max_batches----------------------------------------------------------',max_batches)
        print('max_epochs----------------------------------------------------------',max_epochs)

#    seed = int(time.time())
#    torch.manual_seed(seed)
    global device
    device = torch.device("cpu")

    global model
    model = Darknet(cfgfile)
    model.load_weights(weightfile)
#    print('darknet53',model)

    nsamples = file_lines(trainlist)
    #initialize the model
    if FLAGS.reset:
        model.seen = 0
        init_epoch = 0
    else:
        init_epoch = model.seen//nsamples
    global loss_layers
    loss_layers = model.loss_layers
    for l in loss_layers:
        l.seen = model.seen

    globals()["test_loader"] = load_testlist(testlist)

    params_dict = dict(model.named_parameters())
    params = []
    for key, value in params_dict.items():
        if key.find('.bn') >= 0 or key.find('.bias') >= 0:
            params += [{'params': [value], 'weight_decay': 0.0}]
        else:
            params += [{'params': [value], 'weight_decay': decay*batch_size}]
    global optimizer
    optimizer = optim.SGD(model.parameters(), 
                        lr=learning_rate/batch_size, momentum=momentum, 
                        dampening=0, weight_decay=decay*batch_size)

    if evaluate:
        logging('evaluating ...')
        test(0)
    else:
        try:
            print("Training for ({:d},{:d})".format(init_epoch, max_epochs))
            fscore = 0
            for epoch in range(init_epoch, max_epochs):
                print('epoch=================',epoch)
                # 训练
                nsamples = train(epoch)
                # 训练超过10次后，每训练一次，测试一次
#                if epoch > save_interval:
#                    fscore = test(epoch)
#                fscore = test(epoch)
                # 间隔10次保存一次模型
#                if (epoch+1) % save_interval == 0:
#                    savemodel(epoch, nsamples)
                savemodel(epoch, nsamples)
                # # #
                if FLAGS.localmax and fscore > mfscore:
                    mfscore = fscore
                    savemodel(epoch, nsamples, True)
                print('-'*90)
        except KeyboardInterrupt:
            print('='*80)
            print('Exiting from training by interrupt')

# 调整学习率                
def adjust_learning_rate(optimizer, batch):
    print('adjust_learning_rate input batch',batch)
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate
    for i in range(len(steps)):
        scale = scales[i] if i < len(scales) else 1
        if batch >= steps[i]:
            lr = lr * scale
            if batch == steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr/batch_size
    print('adjust_learning_rate output lr',lr)
    return lr
# 训练
def train(epoch):
    print(train)
    global processed_batches
    t0 = time.time()

    init_width = model.width
    init_height = model.height
    # 加载训练数据
    train_loader = torch.utils.data.DataLoader(
        utils.listDataset(trainlist, shape=(init_width, init_height),
                        shuffle=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                        ]), 
                        train=True, 
                        seen=model.seen,
                        batch_size=batch_size,
                        num_workers=num_workers),
                        batch_size=batch_size, shuffle=False)
    print('train data',train_loader)
    processed_batches = model.seen//batch_size
    # 调整学习率
    lr = adjust_learning_rate(optimizer, processed_batches)
    logging('epoch %d, processed %d samples, lr %e' % (epoch, epoch * len(train_loader.dataset), lr))
    # 切换到训练模式
    model.train()
    avg_time = torch.zeros(9)
    for batch_idx, (data, target) in enumerate(train_loader):
        print('train data split data',data.shape)
        print('train data split target',target.shape)
        adjust_learning_rate(optimizer, processed_batches)
        processed_batches = processed_batches + 1
        data, target = data.to(device), target.to(device)

        # 梯度清零
        optimizer.zero_grad()

        # 前向传播
        output = model.forward(data)
        
        # 输出一个字典 dict{0:{'x':13x13,'a':anchor,'n':number},
        #                  1:{'x':26x26,'a':anchor,'n':},
        #                  2:{'x':52x52,'a':anchor,'n':}}

        # 损失函数(损失函数重命名,loss_layers层返回枚举 用loss代替 l)
        org_loss = []
        for i, l in enumerate(loss_layers):
            l.seen = l.seen + data.data.size(0)
            # 分别对 13x13,26x26,52x52计算损失函数
            ol=l(output[i]['x'], target)
            org_loss.append(ol)
        # 反向传播
        sum(org_loss).backward()
        print('backward',batch_idx,sum(org_loss))
        
        # 截断
        nn.utils.clip_grad_norm_(model.parameters(), 1000)
        # 梯度更新
        optimizer.step()
        del data, target
        org_loss.clear()
        gc.collect()
    print('---')
    t1 = time.time()
    nsamples = len(train_loader.dataset)
    logging('training with %f samples/s' % (nsamples/(t1-t0)))
    return nsamples
# 保存模型
def savemodel(epoch, nsamples, curmax=False):
    print('save model start')
    if curmax:
        logging('save local maximum weights to %s/localmax.weights' % (backupdir))
    else:
        logging('save weights to %s/%06d.weights' % (backupdir, epoch+1))
    model.seen = (epoch + 1) * nsamples
    if curmax: 
        model.save_weights('%s/localmax.weights' % (backupdir))
    else:
        model.save_weights('%s/%06d.weights' % (backupdir, epoch+1))
        old_wgts = '%s/%06d.weights' % (backupdir, epoch+1-keep_backup*save_interval)
        try: #  it avoids the unnecessary call to os.path.exists()
            os.remove(old_wgts)
        except OSError:
            pass
    print('save model end')
# 测试
def test(epoch):
    print('start test',test)
    def truths_length(truths):
        for i in range(50):
            if truths[i][1] == 0:
                return i
        return 50
    # 切换到测试模式
    model.eval()
#    cur_model = curmodel()
    num_classes = model.num_classes
    total       = 0.0
    proposals   = 0.0
    correct     = 0.0

    with torch.no_grad():
        for data, target in test_loader:
            print('test input:data',data.shape)
            print('test input:target',target.shape)
            data = data.to(device)
            # 前向传播
            output = model.forward(data)
#            print('output',output)
            # bounding box预测结果
            all_boxes = get_all_boxes(output, conf_thresh, num_classes)
#            print('all_boxes***************************',all_boxes)
            for k in range(data.size(0)):
                boxes = all_boxes[k]
                print('boxes**********************************************',np.array(boxes).shape)
                # 极大值抑制
                boxes = np.array(nms(boxes, nms_thresh))
                print('boxes nms',np.array(boxes).shape)
                
                truths = target[k].view(-1, 5)
                num_gts = truths_length(truths)
                total = total + num_gts
                num_pred = len(boxes)
        
                if num_pred == 0:
                    continue

                proposals += int((boxes[:,4]>conf_thresh).sum())
                for i in range(num_gts):
                    gt_boxes = torch.FloatTensor([truths[i][1], truths[i][2], truths[i][3], truths[i][4], 1.0, 1.0, truths[i][0]])
                    gt_boxes = gt_boxes.repeat(num_pred,1).t()
                    pred_boxes = torch.FloatTensor(boxes).t()
                    best_iou, best_j = torch.max(multi_bbox_ious(gt_boxes, pred_boxes, x1y1x2y2=False),0)
                    # pred_boxes and gt_boxes are transposed for torch.max
                    if best_iou > iou_thresh and pred_boxes[6][best_j] == gt_boxes[6][0]:
                        correct += 1
                        
    precision = 1.0*correct/(proposals+eps)
    recall = 1.0*correct/(total+eps)
    fscore = 2.0*precision*recall/(precision+recall+eps)
    logging("correct: %d, precision: %f, recall: %f, fscore: %f" % (correct, precision, recall, fscore))
    return fscore

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d',
        type=str, default='cfg/voc.data', help='data definition file')
    parser.add_argument('--config', '-c',
        type=str, default='cfg/yolov3-voc.cfg', help='network configuration file')
    parser.add_argument('--weights', '-w',
        type=str, default='weights/darknet53.conv.74', help='initial weights file')
    parser.add_argument('--reset', '-r',
        action="store_true", default=True, help='initialize the epoch and model seen value')
    parser.add_argument('--localmax', '-l',
        action="store_true", default=False, help='save net weights for local maximum fscore')

    FLAGS, _ = parser.parse_known_args()
    main()
