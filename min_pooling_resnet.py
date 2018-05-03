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
from torch.nn import functional as F

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


class MinPooling(nn.MaxPool2d):
    def __init__(self,kernel_size=2, stride=2, padding=0):
        super(MinPooling,self).__init__(kernel_size=kernel_size, stride=stride, padding=padding)
        self.negative_mask = Variable(torch.Tensor([-1.]), requires_grad=False)
    def forward(self,x):
        x = torch.mul(x,self.negative_mask.expand_as(x))
        x = F.max_pool2d(x, self.kernel_size, self.stride,
                            self.padding, self.dilation, self.ceil_mode,
                            self.return_indices)
        x = torch.mul(x,self.negative_mask.expand_as(x))
        return x


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


class ResNet_Min_Coarse_Pooling(nn.Module):

    def __init__(self, block, layers, num_fine_classes=100,num_coarse_classes=20):
        self.inplanes = 16
        super(ResNet_Min_Coarse_Pooling, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.minpooling = MinPooling(kernel_size=3, stride=1, padding=1)
        # self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0]) #32

        self.layer2_1_coarse = self._make_layer(block, 128, layers[1], stride=2,inplanes=64*block.expansion)#16
        self.layer3_1_coarse = self._make_layer(block, 256, layers[2], stride=2,inplanes=128*block.expansion)#8
        self.bn2_1_coarse = nn.BatchNorm2d(256*block.expansion)
        self.fc1_coarse = nn.Linear(256*block.expansion, num_coarse_classes)

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,inplanes=64*block.expansion)#16

        self.layer3_2_coarse = self._make_layer(block, 256, layers[2], stride=2,inplanes=128*block.expansion)#8
        self.bn2_2_coarse = nn.BatchNorm2d(256*block.expansion)
        self.fc2_coarse = nn.Linear(256*block.expansion, num_coarse_classes)

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,inplanes=128*block.expansion)#8
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)#4

        self.bn2_3_coarse = nn.BatchNorm2d(256*block.expansion)
        self.fc3_coarse = nn.Linear(256*block.expansion, num_coarse_classes)

        self.bn2 = nn.BatchNorm2d(256*block.expansion)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(256*block.expansion, num_fine_classes)
        # self.dropout = nn.Dropout()
        # self.softmax = nn.Softmax()

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, inplanes=None,stride=1):
        if inplanes == None:
            inplanes = self.inplanes
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            # downsample = nn.Sequential(
            #     nn.Conv2d(self.inplanes, planes * block.expansion,
            #               kernel_size=1, stride=stride),
            #     nn.BatchNorm2d(planes * block.expansion),
            # )

            downsample = nn.Conv2d(inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride)
        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        # self.inplanes = planes * block.expansion
        inplanes = planes*block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))
        self.inplanes = inplanes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        out_coarse_1 = self.minpooling(x)
        out_coarse_1 = self.layer2_1_coarse(out_coarse_1)
        out_coarse_1 = self.layer3_1_coarse(out_coarse_1)
        out_coarse_1 = self.bn2_1_coarse(out_coarse_1)
        out_coarse_1 = self.relu(out_coarse_1)
        out_coarse_1 = self.avgpool(out_coarse_1)
        out_coarse_1 = out_coarse_1.view(out_coarse_1.size(0), -1)
        out_coarse_1 = self.fc1_coarse(out_coarse_1)

        x = self.layer2(x)
        out_coarse_2 = self.minpooling(x)
        out_coarse_2 = self.layer3_2_coarse(out_coarse_2)
        out_coarse_2 = self.bn2_2_coarse(out_coarse_2)
        out_coarse_2 = self.relu(out_coarse_2)
        out_coarse_2 = self.avgpool(out_coarse_2)
        out_coarse_2 = out_coarse_2.view(out_coarse_2.size(0), -1)
        out_coarse_2 = self.fc2_coarse(out_coarse_2)

        x = self.layer3(x)
        out_coarse_3 = self.minpooling(x)
        out_coarse_3 = self.bn2_3_coarse(out_coarse_3)
        out_coarse_3 = self.relu(out_coarse_3)
        out_coarse_3 = self.avgpool(out_coarse_3)
        out_coarse_3 = out_coarse_3.view(out_coarse_3.size(0), -1)
        out_coarse_3 = self.fc3_coarse(out_coarse_3)

        # x = self.layer4(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x,out_coarse_1,out_coarse_3,out_coarse_3

def resnet56_cifar():
    depth = 56
    n=(depth-2)/9
    model = ResNet_Min_Coarse_Pooling(Bottleneck, [n, n, n, n])
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


folder_to_save = './Min_Pooling_coarse2fine_2/'
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

coarse_to_fine_dict = pickle.load(open('./coarse_to_fine_dict.txt','r'))
trainset = CIFAR100_multilabel(root='./datasets',train=True,
                                        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=1)

testset = CIFAR100_multilabel(root='./datasets', train=False,
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=1)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

model = resnet56_cifar().cuda()
# pretrained = torch.load(weights_location)
#PRETRAINED WEIGHTS
# for m1,m0 in zip(model.modules(),pretrained.modules()):
#     if isinstance(m1, nn.Conv2d) or isinstance(m1, nn.BatchNorm2d):
#         m1.weight.data = m0.weight.data
#         m1.requires_grad = False

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
history = {}
# history['coarse'] = {'test_accuracy' : [],'training_accuracy' : []}
history['coarse'] = {str(i) : {'test_accuracy' : [],'training_accuracy' : []} for i in range(3)}
history['fine'] = {'test_accuracy' : [],'training_accuracy' : []}
for epoch in range(nEpoch):  # loop over the dataset multiple times
    # if epoch == 80 or epoch == 120:
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] *= 0.1
    # if epoch == 60:
    #     for m in model.modules():
    #         if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d):
    #             m.requires_grad = True
    print('EPOCH %d')%(epoch)
    running_loss = 0.0
    trainloader = tqdm(trainloader)
    correct_training_coarse = 3*[0.]
    correct_training_fine = 0.
    # correct_training_overall_fine = 0.
    total = 0.
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs = Variable(data[0]).cuda()
        fine_label = Variable(data[1]).cuda()
        coarse_label = Variable(data[2]).cuda()
        # wrap them in Variable
        # inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs[0], fine_label)
        _, predicted_fine = torch.max(outputs[0].data, 1)
        correct_training_fine += (predicted_fine == fine_label.data).sum()
        for k,coarse_output in enumerate(outputs[1::]):
            loss += criterion(coarse_output, coarse_label)
            _, predicted_coarse = torch.max(coarse_output.data, 1)
            correct_training_coarse[k] += (predicted_coarse == coarse_label.data).sum()

        total += inputs.size(0)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss = 0.99 * running_loss + 0.01 * loss.data[0]
        trainloader.set_postfix(loss=running_loss)
    for coarse in range(3):
        history['coarse'][str(coarse)]['training_accuracy'] += [correct_training_coarse[coarse]/total]
    history['fine']['training_accuracy'] +=[correct_training_fine/total]
    # for k in range(20):
    #     history['fine'][str(k)]['training_accuracy'] += [correct_training_fine[k]/total]

    print('TRAINING ACCURACY FINE: %.3f')%(history['fine']['training_accuracy'][-1])
    for k in range(3):
        print('TRAINING ACCURACY COARSE %d : %.3f')%(k,history['coarse'][str(k)]['training_accuracy'][-1])
    # print('TRAINING ACCURACY OVERALL FINE: %.3f')%(history['overall']['training_accuracy'][-1])
    correct_tesitng_coarse = 3*[0.]
    total = 0.
    correct_testing_fine = 0.
    for data in testloader:
        inputs = Variable(data[0]).cuda()
        fine_label = Variable(data[1]).cuda()
        coarse_label = Variable(data[2]).long().cuda()
        outputs = model(inputs.cuda())
        _, predicted_fine = torch.max(outputs[0].data, 1)
        correct_testing_fine += (predicted_fine == fine_label.data).sum()

        for k,coarse_output in enumerate(outputs[1::]):
            _, predicted_coarse = torch.max(coarse_output.data, 1)
            correct_tesitng_coarse[k] += (predicted_coarse == coarse_label.data).sum()

        total += inputs.size(0)

    history['fine']['test_accuracy'] += [correct_testing_fine/total]
    for coarse in range(3):
        history['coarse'][str(coarse)]['test_accuracy'] += [correct_tesitng_coarse[coarse]/total]

    print('TEST ACCURACY FINE: %.3f')%(history['fine']['test_accuracy'][-1])
    for k in range(3):
        print('TEST ACCURACY COARSE %d : %.3f')%(k,history['coarse'][str(k)]['test_accuracy'][-1])

    torch.save(model,folder_to_save+'model.pth.tar')

with open(folder_to_save + 'history.txt','w') as fp:
    pickle.dump(history,fp)
print('**** Finished Training ****')
