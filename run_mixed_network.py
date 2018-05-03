import torch
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable
import torch.nn as nn

import torch.optim as optim

from tqdm import tqdm
import random
import cPickle as pickle
import shutil
import sys
import os
import numpy as np
import math
import os
from PIL import Image

import torch.nn as nn
import math
from nn_utils import *
import torch.utils.model_zoo as model_zoo

def resnet56_cifar():
    depth = 56
    n=(depth-2)/9
    model = ResNet_Full_Coarse_Auxs(Bottleneck, [n, n, n, n])
    return model

def calculate_accuracy(model,outputs,labels):
    _, predicted = torch.max(outputs.data, 1)
    total = float(labels.size(0))
    correct = (predicted == labels.data).sum()
    return correct/total

def multi_gpu(model,inputs,gpus):
    replicas = nn.parallel.replicate(model, gpus)#gpus = [0,1,2]
    inputs = nn.parallel.scatter(inputs, gpus)
    outputs = nn.parallel.parallel_apply(replicas, inputs)
    return nn.parallel.gather(outputs, [0])

def freeze_layers(model,layers_name,defreeze=False,toogle=False):
    layers_dict = model.state_dict()
    if toogle == False:
        for layer in layers_name:
            if layer+'.weight' in layers_dict.keys():
                layers_dict[layer+'.weight'].requires_grad = defreeze
            if layer+'.bias' in layers_dict.keys():
                layers_dict[layer+'.bias'].requires_grad = defreeze
            if layer+'.running_mean' in layers_dict.keys():
                layers_dict[layer+'.running_mean'].requires_grad = defreeze
            if layer+'.running_var' in layers_dict.keys():
                layers_dict[layer+'.running_var'].requires_grad = defreeze
    if toogle == True:
        for layer in layers_dict.keys():
            name = layer.split('.')[0]
            if name not in layers_name:
                layers_dict[layer].requires_grad = defreeze

folder_to_save = './Run4_full_outputs_coarse/'
torch.backends.cudnn.enabled = True
if not os.path.isdir(folder_to_save):
    os.mkdir(folder_to_save)
# if not os.path.isfile(folder_to_save+'run_file.py'):
shutil.copyfile('./run_mixed_network.py',folder_to_save+'run_file.py')
nEpoch = 150
depth = 27
filters = 64
gpus = [0,1]
depth_per_block = np.floor(depth/3.).astype(int)
depth_res = np.mod(depth,3)
num_filters = filters*np.asarray([1,2,4])
torch.set_default_tensor_type('torch.cuda.FloatTensor')
if not os.path.isdir(folder_to_save):
    os.mkdir(folder_to_save)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
trainset = CIFAR100_multilabel(root='./datasets', train=True,
                                        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=1)

testset = CIFAR100_multilabel(root='./datasets', train=False,
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=1)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
num_coarse_outputs = 4
model = resnet56_cifar().cuda()
# model = torch.nn.DataParallel(model, gpus)
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(),lr=0.1,momentum=0.9,weight_decay=0.0001)
optimizer = optim.Adam(model.parameters())
history = {}
history['acc_fine_test'] = list()
history['acc_fine_training'] = list()

for i in range(num_coarse_outputs):
    history['acc_coarse_test'+str(i)] = list()
    history['acc_coarse_training'+str(i)] = list()

# aux_layers = ['aux_bn','aux_fc1','aux_fc2','aux_fc3']

for epoch in range(nEpoch):  # loop over the dataset multiple times
    if isinstance(optimizer,optim.SGD):
        if epoch == 80 or epoch == 120:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
    print('EPOCH %d')%(epoch)
    running_loss = 0.0
    trainloader = tqdm(trainloader)
    acc_fine = list()
    acc_coarse = {str(i) : [] for i in range(num_coarse_outputs)}
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels_fine,labels_coarse = data

        # wrap them in Variable
        inputs, labels_fine,labels_coarse = Variable(inputs).cuda(), Variable(labels_fine).cuda(), Variable(labels_coarse).cuda()

        # zero the parameter gradients
        optimizer.zero_grad()
        # freeze_layers(model,aux_layers)

        # forward + backward + optimize

        outputs = model(inputs)
        # outputs = multi_gpu(model,inputs,gpus)
        acc_fine += [calculate_accuracy(model,outputs[0],labels_fine)]
        for k,out in enumerate(outputs[1::]):
            acc_coarse[str(k)] += [calculate_accuracy(model,out,labels_coarse)]

        # freeze_layers(model,aux_layers,defreeze=True,toogle=True)
        # freeze_layers(model,aux_layers)
        loss = 0.5*criterion(outputs[0], labels_fine)
        # freeze_layers(model,aux_layers,defreeze=False,toogle=True)
        # freeze_layers(model,aux_layers,defreeze=True)
        for output in outputs[1::]:
            loss += 0.5*criterion(output,labels_coarse)

        # loss1.backward(retain_variables=True)
        loss.backward()
        # os.system('nvidia-smi')
        optimizer.step()

        # print statistics
        running_loss = 0.99 * running_loss + 0.01 * loss.data[0]
        trainloader.set_postfix(loss=running_loss)
    history['acc_fine_training'].append(np.mean(acc_fine))
    for i in range(len(acc_coarse.keys())):
        history['acc_coarse_training' + str(i)].append(np.mean(acc_coarse[str(i)]))
    correct_fine = 0.
    correct_coarse = num_coarse_outputs*[0.]
    total = 0.
    for data in testloader:
        inputs, labels_fine,labels_coarse = data
        # wrap them in Variable
        inputs, labels_fine,labels_coarse = Variable(inputs).cuda(), Variable(labels_fine).cuda(), Variable(labels_coarse).cuda()

        outputs = model(inputs)
        _, predicted_fine = torch.max(outputs[0].data, 1)
        # predicted_coarse = []
        for i,out in enumerate(outputs[1::]):
            predicted_coarse = torch.max(out.data, 1)[1]
            correct_coarse[i] += (predicted_coarse == labels_coarse.data).sum()
        total += labels_fine.size(0)
        correct_fine += (predicted_fine == labels_fine.data).sum()
        # correct_coarse += (predicted_coarse == labels_coarse.data).sum()
    history['acc_fine_test'].append(correct_fine/total)
    for i,x in enumerate(correct_coarse):
        history['acc_coarse_test'+str(i)].append(x/total)

    print('TRAINING ACCURACY FINE : %.3f')%(history['acc_fine_training'][-1])
    print('TEST ACCURACY FINE : %.3f')%(history['acc_fine_test'][-1])

    for i,x in enumerate(correct_coarse):
        print('TRAINING ACCURACY COARSE %d : %.3f')%(i,history['acc_coarse_training'+str(i)][-1])
        print('TEST ACCURACY COARSE %d : %.3f')%(i,history['acc_coarse_test'+str(i)][-1])
with open(folder_to_save + 'history_'+'.txt','w') as fp:
    pickle.dump(history,fp)
print('**** Finished Training ****')
