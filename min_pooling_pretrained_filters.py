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


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=100):

        super(ResNet, self).__init__()
        self.inplanes = 16
        self.block = block
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
    def forward_1(self,x):
        x = self.conv1(x)
        x = self.layer1(x)
        return x
    def forward_2(self,x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x
    def forward_3(self,x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x



    def get_hook_value(self):
        return self.hook_out
    def hook_at_layer(self,layer_name):
        layer = self._modules.get(layer_name)
        # self.hook_out = torch.zeros((256*self.block.expansion,1))
        def fun(m, i, o):
            self.hook_out = o
        self.hook = layer.register_forward_hook(fun)
    def remove_hook(self):
        self.hook.remove()

class MLP_Min_Pooling(nn.Module):

    def __init__(self, in_shape,out_dim):
        super(MLP_Min_Pooling, self).__init__()
        # self.minpooling = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv = nn.Conv2d(in_shape[0], in_shape[0]/16,kernel_size=3, stride=1,padding=1)
        self.bn = nn.BatchNorm2d(in_shape[0]/16)
        in_dim = np.prod(in_shape[1::])*in_shape[0]/16
        if in_dim > out_dim:
            self.fc1 = nn.Linear(in_dim,in_dim/2)
            self.fc2 = nn.Linear(in_dim/2,in_dim/4)
            self.fc_out = nn.Linear(in_dim/4,out_dim)
        else:
            self.fc1 = nn.Linear(in_dim,out_dim/4)
            self.fc2 = nn.Linear(out_dim/4,out_dim/2)
            self.fc_out = nn.Linear(out_dim/2,out_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout()

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.minpooling(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc_out(x)
        return x

class Resnet_Min_Pooling(nn.Module):

    def __init__(self, in_dim,out_dim):
        super(MLP_Min_Pooling, self).__init__()
        self.minpooling = MinPooling(kernel_size=2, stride=2, padding=0)

        if in_dim/4 > out_dim:
            self.fc1 = nn.Linear(in_dim,in_dim/2)
            self.fc2 = nn.Linear(in_dim/2,in_dim/3)
            self.fc_out = nn.Linear(in_dim/64,out_dim)
        else:
            self.fc1 = nn.Linear(in_dim,out_dim/4)
            self.fc2 = nn.Linear(out_dim/4,out_dim/2)
            self.fc_out = nn.Linear(out_dim/2,out_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout()

        self.init_weights()

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

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # x = self.minpooling(x)
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc_out(x)


def resnet56_cifar():
    depth = 56
    n=(depth-2)/9
    model = ResNet(Bottleneck, [n, n, n, n])
    return model
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
folder_to_save = './Pretrained_Filters_None_Pooling_Run1_wd_5/'
weights_location = './Baseline3_100/model.pth.tar'
torch.backends.cudnn.enabled = True
print(folder_to_save)
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
trainset = CIFAR100_multilabel(root='./datasets',train=True,
                                        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=1)

testset = CIFAR100_multilabel(root='./datasets', train=False,
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=1)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

model = torch.load(weights_location)
model.block = Bottleneck
# model = resnet56_cifar().cuda()

resnet_score = test_model(model,testloader,[0,1])

print('TEST ACCURACY PRETRAINED RESNET FINE: %.3f')%(resnet_score)

layer_names = ['layer1','layer2','layer3']

model_output_functions = [model.forward_1,model.forward_2,model.forward_3]
layer_out_shapes = [(64*model.block.expansion,32,32),(128*model.block.expansion,16,16),(256*model.block.expansion,8,8)]
if os.path.isfile(folder_to_save+'history.txt'):
    history = pickle.load(open(folder_to_save + 'history.txt','r'))
    del history['layer3']
else:
    history = {}
for layer_name,layer_out_shape,model_output_function in zip(layer_names,layer_out_shapes,model_output_functions):
    if layer_name not in history:
    # model.hook_at_layer(layer_name)
        mlp = MLP_Min_Pooling(in_shape=layer_out_shape,out_dim=20).cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(mlp.parameters(),weight_decay=10.e-5)
        history[layer_name] = {'test_accuracy' : [],'training_accuracy' : []}
        for epoch in range(nEpoch):  # loop over the dataset multiple times
            print('EPOCH %d')%(epoch)
            running_loss = 0.0
            trainloader = tqdm(trainloader)
            # correct_training_overall_fine = 0.
            total = 0.
            correct_training_coarse = 0.
            for i, data in enumerate(trainloader, 0):
                # get the inputs
                inputs = Variable(data[0]).cuda()
                # fine_label = Variable(data[1]).cuda()
                coarse_label = Variable(data[2]).cuda()
                # wrap them in Variable
                # inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                out = model_output_function(inputs)
                output_coarse = mlp(out.cuda())

                loss = criterion(output_coarse, coarse_label)
                loss.backward()
                optimizer.step()

                _, predicted_coarse = torch.max(output_coarse.data, 1)
                correct_training_coarse += (predicted_coarse == coarse_label.data).sum()

                total += inputs.size(0)


                # print statistics
                running_loss = 0.99 * running_loss + 0.01 * loss.data[0]
                trainloader.set_postfix(loss=running_loss)
            history[layer_name]['training_accuracy'] += [correct_training_coarse/total]

            print('TRAINING ACCURACY COARSE: %.3f')%(history[layer_name]['training_accuracy'][-1])

            total = 0.
            correct_testing_coarse = 0.
            for data in testloader:
                inputs = Variable(data[0]).cuda()
                # fine_label = Variable(data[1]).cuda()
                coarse_label = Variable(data[2]).long().cuda()
                out = model_output_function(inputs.cuda())
                output_coarse = mlp(out.cuda())

                _, output_coarse = torch.max(output_coarse.data, 1)
                correct_testing_coarse += (output_coarse == coarse_label.data).sum()

                total += inputs.size(0)

            history[layer_name]['test_accuracy'] += [correct_testing_coarse/total]

            print('TEST ACCURACY COARSE: %.3f')%(history[layer_name]['test_accuracy'][-1])
            torch.save(mlp,folder_to_save+'mlp' +layer_name+'.pth.tar')
            with open(folder_to_save + 'history.txt','w') as fp:
                pickle.dump(history,fp)

        print('**** Finished Training ****')
    else:
        print('SKIPPING %s')%(layer_name)
