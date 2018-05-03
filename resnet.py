import torch
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable
import torch.nn as nn

import torch.optim as optim

from tqdm import tqdm
import random
import cPickle as pickle
import sys
import os
import numpy as np
import math

from torch.nn.init import kaiming_normal
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from nn_utils import CIFAR100_multilabel

def get_coarse_to_fine_dictionary(coarse_labels,fine_labels):
    coarse_labels_to_fine = dict()
    for current_label in range(np.max(coarse_labels)+1):
        values_in_coarse_label = list()
        for coarse_label,fine_label in zip(coarse_labels,fine_labels):
            if(int(coarse_label) == int(current_label)) and (fine_label not in values_in_coarse_label):
                values_in_coarse_label.append(fine_label)
        coarse_labels_to_fine[str(current_label)] = np.asarray(values_in_coarse_label)
    return coarse_labels_to_fine

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        # we do pre-activation
        out = self.relu(self.bn1(x))
        out = self.conv1(out)

        out = self.relu(self.bn2(out))
        out = self.conv2(out)

        out = self.relu(self.bn3(out))
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=100):
        self.inplanes = 16
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0]) #32
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)#16
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)#8
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)#4
        self.bn2 = nn.BatchNorm2d(256*block.expansion)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(256*block.expansion, num_classes)
        self.dropout = nn.Dropout()
        # self.softmax = nn.Softmax()

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # downsample = nn.Sequential(
            #     nn.Conv2d(self.inplanes, planes * block.expansion,
            #               kernel_size=1, stride=stride),
            #     nn.BatchNorm2d(planes * block.expansion),
            # )

            downsample = nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet56_cifar():
    depth = 56
    n=(depth-2)/9
    model = ResNet(Bottleneck, [n, n, n, n],num_classes=20)
    return model

def get_overall_accuracy(predicted_values,predicted_labels,overall_fines,coarse_to_fine_dict):
    overall_prediction = np.zeros((overall_fines.size(0),)).astype(int)
    for j in range(overall_fines.size(0)):
        max_value = 0.
        max_fine = None
        for k,(predicted_value,predicted_label) in enumerate(zip(predicted_values,predicted_labels)):
            if predicted_value[j] > max_value:
                max_value = predicted_value[j]
                max_fine = predicted_label[j]
                coarse_class = k

        overall_prediction[j] = coarse_to_fine_dict[str(coarse_class)][max_fine-1].astype(int)
    overall_prediction = torch.from_numpy(overall_prediction).long().cuda()

    overall_correct = (overall_prediction == overall_fines.data).sum()

    return overall_correct

def test_model(model,data,out_indexes):
    total = 0.
    score = 0.
    for test_inputs in testloader:
        inputs = Variable(test_inputs[out_indexes[0]]).cuda()
        labels = Variable(test_inputs[out_indexes[1]]).long().cuda()
        out = model(inputs.cuda())

        _, out = torch.max(out.data, 1)
        score += (out == labels.data).sum()

        total += inputs.size(0)
    return score/total

def train_epoch(model,data,optimizer,criterion,out_indexes):
    total = 0.
    correct = 0.
    running_loss = 0.
    for i,x  in enumerate(data, 0):
        # get the inputs
        inputs = Variable(x[out_indexes[0]]).cuda()
        labels = Variable(x[out_indexes[1]]).cuda()
        optimizer.zero_grad()

        out = model(inputs)
        loss = criterion(out, labels)
        _, predicted = torch.max(out.data, 1)
        correct += (predicted == labels.data).sum()
        total += inputs.size(0)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss = 0.99 * running_loss + 0.01 * loss.data[0]
        data.set_postfix(loss=running_loss)
    return correct/total


folder_to_save = './Baseline3_20/'
print(folder_to_save)
# weights_location = './Baseline_coarse/model.pth.tar'
torch.backends.cudnn.enabled = True

nEpoch = 150
depth = 27
filters = 64
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

# coarse_to_fine_dict = pickle.load(open('./coarse_to_fine_dict.txt','r'))
trainset = CIFAR100_multilabel(root='./datasets', train=True,
                                        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=1)

testset = CIFAR100_multilabel(root='./datasets',  train=False,
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=1)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

model = resnet56_cifar().cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=0.05,momentum=0.9,weight_decay=10.e-4)
history = {}
history['fine'] = {'test_accuracy' : [],'training_accuracy' : []}

for epoch in range(nEpoch):  # loop over the dataset multiple times
    if epoch == 80 or epoch == 120:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
    print('EPOCH %d')%(epoch)
    running_loss = 0.0
    trainloader = tqdm(trainloader)

    history['fine']['training_accuracy'] +=[train_epoch(model,trainloader,optimizer,criterion,[0,2])]
    print('TRAINING ACCURACY FINE: %.3f')%(history['fine']['training_accuracy'][-1])

    history['fine']['test_accuracy'] += [test_model(model,testloader,[0,2])]
    print('TEST ACCURACY FINE: %.3f')%(history['fine']['test_accuracy'][-1])
    torch.save(model,folder_to_save+'model.pth.tar')

with open(folder_to_save + 'history.txt','w') as fp:
    pickle.dump(history,fp)
print('**** Finished Training ****')
