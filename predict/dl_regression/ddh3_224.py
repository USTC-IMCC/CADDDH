#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 18:05:24 2018

@author: liuchuanbin
"""

from __future__ import print_function, division
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
import time
from torchvision import transforms, utils, datasets, models
from PIL import Image
import pandas as pd
from skimage import io, transform
import warnings
warnings.filterwarnings("ignore")
from vggmodel3_224 import *

parser = argparse.ArgumentParser(description='Bone age asessment')
parser.add_argument('--lr', default='0.1', type=float, help='learning rate')
parser.add_argument('--it', default='00', type=str, help='introduction')
parser.add_argument('--lo', default='0', type=int, help='loss function 0:L1 1:SmoothL1 2: L2')
parser.add_argument('--nm', default='0', type=int, help='network modle 0:vgg11; 1:vgg11_bn; 2:vgg16; 3:vgg16_bn')
parser.add_argument('--bs', default='16', type=int, help='batch size')
args = parser.parse_args()

class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        #print((image.shape))

        image = np.array([image,image,image])
        #print((image.shape))

        image = image.transpose((1, 2, 0))
        landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}
class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}

class CenterCrop(object):
    """Crop center of the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = int((h - new_h)/2)
        left =int((w - new_w)/2)

        #image = image[top: top + new_h,
        #              left: left + new_w]

        image = image[top: top + new_h,
                      left: left + new_w]
        #image = image[::]
        
        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f 



modellist = [vgg11(),vgg11_bn(),vgg16(),vgg16_bn(),vgg19(),vgg19_bn()]
#model = model.double()
model = modellist[args.nm]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
print ('==> Building model..')
print(model)

lr = args.lr
bs = args.bs
introudction = args.it
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=0.0001)

criterion1 = nn.MSELoss()
criterion2 = nn.MSELoss(size_average=False)
criterion3 = nn.L1Loss()
criterion4 = nn.L1Loss(size_average=False)
criterions = [nn.L1Loss(),nn.SmoothL1Loss(),nn.MSELoss()]
criterion = criterions[args.lo]
total_epoch = 300

transformed_traindataset = FaceLandmarksDataset(csv_file='/gdata/liucb/DDH/train/file.csv',
                                           root_dir='/gdata/liucb/DDH/train/original_jpg/',
                                           transform=transforms.Compose([
                                               Rescale(256),
                                               #RandomCrop(288),
                                               #Rescale(224),
                                               #CenterCrop(320),
                                               RandomCrop(224),
                                               #Rescale(224),
                                               ToTensor()
                                           ]))

train_loader = DataLoader(transformed_traindataset, batch_size=bs, shuffle=True)

transformed_testdataset = FaceLandmarksDataset(csv_file='/gdata/liucb/DDH/test/file.csv',
                                           root_dir='/gdata/liucb/DDH/test/original_jpg/',
                                           transform=transforms.Compose([
                                               Rescale(224),
                                               #RandomCrop(288),
                                               #Rescale(224),
                                               #CenterCrop(320),
                                               #RandomCrop(224),
                                               #Rescale(224),
                                               ToTensor()
                                           ]))

test_loader = DataLoader(transformed_testdataset, batch_size=50, shuffle=True)

train_batch_num=len(train_loader)
print ('\nLength of batch per epoch is %d \n'%(len(train_loader)))

def train(epoch):
    start_epocht=time.time()
    model.train()
    print('\nTrain Epoch: %d' % epoch)
    for i, sample_batched in enumerate(train_loader):
        model.train()
        images = Variable(sample_batched['image'].float().to(device))
        labels = Variable(sample_batched['landmarks'].float().to(device))
        labels = labels.view(-1,12)
        outputs = model(images)

        optimizer.zero_grad()
        outputs = model(images)
#        outputs = outputs.resize(torch.numel(outputs))
        loss = criterion(outputs, labels)
        rmseloss = criterion1(outputs, labels)**0.5
        l1loss = criterion3(outputs, labels)
        loss.backward()
        optimizer.step()
        costt = format_time(time.time()-startt)

        if i % 1 == 0:
            print('[%10s] Run Time [%4d/%4d] Batch in [%3d/%3d] Epoch, Loss is %12.6f, RMSE is %12.6f, MAE is %12.6f'%(costt,i,train_batch_num,epoch,total_epoch,(loss.cpu().detach().numpy()),(rmseloss.cpu().detach().numpy()),(l1loss.cpu().detach().numpy())))
        #costt = format_time(time.time()-startt)
        #print('Cost time',costt)
        #if i==3:
        #    break

        if i % 30 == 29:
            test(epoch)
#   if i==1:
#            break
    epochcostt=format_time(time.time()-start_epocht)
    print('==>Epoch cost time',epochcostt)

    '''
    for i, (images, labels) in enumerate(data_loader):
        images = Variable(images.to(device))
        labels = Variable(labels.float().to(device))
        optimizer.zero_grad()
        outputs = model(images)
        outputs = outputs.resize(torch.numel(outputs))
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
#        print('The %d/%d Batch in %d Epoch,MSE is %.8f'%(i,train_batch_num,epoch,(loss.detach().numpy())))
        print ('The L1 Loss is :%.8f'%(loss.cpu().detach().numpy()))
  '''

def test(epoch):
    print('\n==>Test Epoch: %d' % epoch)
    model.eval()
    rmse_loss = 0.0
    l1_loss = 0.0
    rmseloss = 0.0
    l1loss = 0.0
    with torch.no_grad():
        for i, sample_batched in enumerate(test_loader):
        #images = Variable(images)
        #labels = Variable(labels.float())
            images = Variable(sample_batched['image'].float().to(device))
            labels = Variable(sample_batched['landmarks'].float().to(device))
            labels = labels.view(-1,12)
            outputs = model(images)
            #images = Variable(images.to(device))
            #labels = Variable(labels.float().to(device))
            optimizer.zero_grad()
            #outputs = model(images)
            #outputs = outputs.resize(torch.numel(outputs))
            loss1 = criterion2(outputs, labels)
            loss2 = criterion4(outputs, labels)
            rmse_loss = loss1+rmse_loss
            l1_loss = loss2+l1_loss
        rmseloss = (rmse_loss/(131*12))**0.5
        l1loss = (l1_loss/(131*12))
        print ('Test : RMSE is %.6f, MAE is %.6f'%(float(rmseloss),float(l1loss)))
        '''
        if epoch % 10 ==0:
            print ('==>Saving Model...')
            if not os.path.isdir('checkpoints'):
                os.mkdir('checkpoints')
            ModelName='./checkpoints/'+str(introudction)+'_Female_BAANet_'+str(epoch)+'_Epoch.pkl'
            torch.save(model.state_dict(),ModelName)
        '''
        global acc
        if (rmseloss) < acc:
            print ('==>Saving Best Model...')
            acc = rmseloss
            if not os.path.isdir('checkpoints'):
                os.mkdir('checkpoints')
            ModelName='./checkpoints/'+str(introudction)+'_DDH_Best.pkl'
            torch.save(model.state_dict(),ModelName)
        if not os.path.isdir('state'):
            os.mkdir('state')
        costt = format_time(time.time()-startt)
        filename = './state/'+str(introudction)+'.txt'
        f = open(filename,'w')
        f.write('Epoch ['+str(epoch)+'/'+str(total_epoch)+'],RMSE '+str(acc)+', Runtime '+str(costt)+'\n')
        f.close()


startt = time.time()
acc = 1000
for epoch in range(total_epoch):
#    print ('Best ACC is ',acc)
    train(epoch)
#    test(epoch)
#    break
    if epoch % 50 == 49:
        lr /= 2
        print('==>Changing Learning Rate as %.8f'%lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
