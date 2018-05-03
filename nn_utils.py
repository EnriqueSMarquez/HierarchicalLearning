import torch
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable
import torch.nn as nn

import torch.optim as optim
import numpy as np
import cPickle as pickle
import os
import sys
from PIL import Image

class CIFAR100_multilabel(torchvision.datasets.CIFAR100):
    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False,repetitions_fine=1,repetitions_coarse=1):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.repetitions_fine = repetitions_fine
        self.repetitions_coarse = repetitions_coarse

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_fine_labels = []
            self.train_coarse_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.train_data.append(entry['data'])
                if 'fine_labels' in entry:
                    self.train_fine_labels += entry['fine_labels']
                if 'coarse_labels' in entry:
                    self.train_coarse_labels += entry['coarse_labels']
                fo.close()

            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((50000, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            if 'fine_labels' in entry:
                self.test_fine_labels = entry['fine_labels']
            if 'coarse_labels' in entry:
                self.test_coarse_labels = entry['coarse_labels']
            fo.close()
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target_fine,target_coarse = self.train_data[index], self.train_fine_labels[index],self.train_coarse_labels[index]
        else:
            img, target_fine,target_coarse = self.test_data[index], self.test_fine_labels[index],self.test_coarse_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target_fine = self.target_transform(target_fine)
            target_coarse = self.target_transform(target_coarse)
        output = []
        output += [img]
        for i in range(self.repetitions_fine):
            output += [target_fine]
        for i in range(self.repetitions_coarse):
            output += [target_coarse]

        return output

class CIFAR100_multi_fine_class(torchvision.datasets.CIFAR100):
    def __init__(self, root,coarse_to_fine_dict, train=True,
                 transform=None, target_transform=None,
                 download=False):
        self.root = os.path.expanduser(root)
        self.coarse_to_fine_dict = coarse_to_fine_dict
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_fine_labels = []
            self.train_coarse_labels = []
            self.train_multi_fine_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.train_data.append(entry['data'])
                if 'fine_labels' in entry:
                    self.train_fine_labels += entry['fine_labels']
                if 'coarse_labels' in entry:
                    self.train_coarse_labels += entry['coarse_labels']
                fo.close()

            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((50000, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
            self.train_multi_fine_labels = np.zeros((len(self.train_data),np.max(self.train_coarse_labels)+1,1))

            for i,(image,coarse_label,fine_label) in enumerate(zip(self.train_data,self.train_coarse_labels,self.train_fine_labels)):
                self.train_multi_fine_labels[i,coarse_label,:] = self.coarse_to_fine_dict[str(coarse_label)].tolist().index(fine_label)+1
            pass
        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            if 'fine_labels' in entry:
                self.test_fine_labels = entry['fine_labels']
            if 'coarse_labels' in entry:
                self.test_coarse_labels = entry['coarse_labels']
            fo.close()
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC
            # self.coarse_to_fine_dict = get_coarse_to_fine_dictionary(self.train_coarse_labels,self.train_fine_labels)
            self.test_multi_fine_labels = np.zeros((len(self.test_data),np.max(self.test_coarse_labels)+1,1))

            for i,(image,coarse_label,fine_label) in enumerate(zip(self.test_data,self.test_coarse_labels,self.test_fine_labels)):
                self.test_multi_fine_labels[i,coarse_label,:] = self.coarse_to_fine_dict[str(coarse_label)].tolist().index(fine_label)+1


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target_coarse,target_multi_fine,fine_label = self.train_data[index], self.train_coarse_labels[index],self.train_multi_fine_labels[index],self.train_fine_labels[index]
        else:
            img, target_coarse,target_multi_fine,fine_label = self.test_data[index], self.test_coarse_labels[index],self.test_multi_fine_labels[index],self.test_fine_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target_coarse = self.target_transform(target_coarse)
            fine_label = self.target_transform(fine_label)
            fine_targets = []
            for target in target_multi_fine.flatten().tolist():
                fine_targets += [self.target_transform(target)]
        else:
            fine_targets = target_multi_fine.flatten().tolist()

        output = []
        output += [img]
        output += [target_coarse]
        output += fine_targets
        output += [fine_label]

        return output


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


class ResNet_Single_Aux(nn.Module):

    def __init__(self, block, layers, num_classes=100,coarse_aux=20):
        self.inplanes = 16
        self.coarse_aux = coarse_aux
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0]) #32
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)#16
        if self.coarse_aux != None:
            self.aux_bn = nn.BatchNorm2d(128*block.expansion)
            self.aux_avg = nn.AvgPool2d(8)
            self.aux_fc1 = nn.Linear(128*block.expansion*4,128)
            self.aux_fc2 = nn.Linear(128,64)
            self.aux_fc3 = nn.Linear(64,self.coarse_aux)
            self.aux_fc4 = nn.Linear(self.coarse_aux,num_classes)
            self.fc1 = nn.Linear(256*block.expansion,512)
            self.fc2 = nn.Linear(512,256)
            self.fc3 = nn.Linear(256, num_classes)
            # self.fc = nn.Linear(256*block.expansion+4*128*block.expansion, num_classes)
        else:
            self.fc = nn.Linear(256*block.expansion, num_classes)



        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)#8
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)#4
        self.dropout = nn.Dropout()
        self.bn2 = nn.BatchNorm2d(256*block.expansion)
        self.avgpool = nn.AvgPool2d(8)
        # self.softmax = nn.Softmax()

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
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
        if self.coarse_aux != None:
            aux_x = self.aux_bn(x)
            aux_x = self.relu(aux_x)
            aux_x = self.aux_avg(aux_x)
            aux_x = aux_x.view(aux_x.size(0),-1)
            aux_x = self.aux_fc1(aux_x)
            aux_x = self.dropout(aux_x)
            aux_x = self.relu(aux_x)
            aux_x = self.aux_fc2(aux_x)
            aux_x = self.dropout(aux_x)
            aux_x = self.relu(aux_x)
            aux_x = self.aux_fc3(aux_x)
            # merge_x = self.dropout(aux_x)
            # merge_x = self.relu(aux_x)
            # merge_x = self.aux_fc4(merge_x)
            # merge_x = self.dropout(merge_x)

            x = self.layer3(x)
            x = self.bn2(x)
            x = self.relu(x)
            x = self.avgpool(x)
            x = x.view(x.size(0),-1)
            x = self.fc1(x)
            x = self.dropout(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.dropout(x)
            x = self.relu(x)
            x = self.fc3(x)
            # x = self.dropout(x)
            # x = torch.mul(x,merge_x)
            # aux_x = self.aux_bn(x)
            # aux_x_avg = self.aux_avg(aux_x)
            # aux_x_avg = aux_x_avg.view(aux_x_avg.size(0),-1)
            # aux_x = self.aux_fc(aux_x_avg)
            # aux_relu = self.relu(aux_x)
            # # x = self.layer4(x)
            # x = x.view(x.size(0), -1)
            # x = torch.cat([x,aux_x_avg],1)
            # x = self.fc(x)
            return x,aux_x
        else:
            # x = self.layer4(x)
            x = self.bn2(x)
            x = self.relu(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

class ResNet_Full_Coarse_Auxs(nn.Module):
    'MAX AUX 3 (32,32,3)'
    def __init__(self, block, layers, num_classes=100,coarse_aux=[20,20,20,20]):
        super(ResNet_Full_Coarse_Auxs, self).__init__()
        self.inplanes = 16
        self.coarse_aux = coarse_aux
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.dropout = nn.Dropout()
        # self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0]) #32

        #FIRST AUX
        self.layer1_aux1 = self._make_layer(block, 64, layers[0],inplanes=16) #32
        self.bn_aux1 = nn.BatchNorm2d(64*block.expansion)
        self.fc_aux1 = nn.Linear(64*block.expansion*4*4,coarse_aux[0])
        self.fc2_aux1 = nn.Linear(coarse_aux[0],num_classes)

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)#16
        #SECOND AUX
        self.layer2_aux2 = self._make_layer(block, 128, layers[1], inplanes=64*block.expansion, stride=2)#16
        self.bn_aux2 = nn.BatchNorm2d(128*block.expansion)
        self.fc_aux2 = nn.Linear(128*block.expansion*2*2,coarse_aux[1])
        self.fc2_aux2 = nn.Linear(coarse_aux[1],num_classes)


        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)#8
        #THIRD AUX
        self.layer3_aux3 = self._make_layer(block, 256, layers[2], inplanes=128*block.expansion, stride=2)
        self.bn_aux3 = nn.BatchNorm2d(256*block.expansion)
        self.fc_aux3 = nn.Linear(256*block.expansion,coarse_aux[2])
        self.fc2_aux3 = nn.Linear(coarse_aux[2],num_classes)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)#4
        self.bn_last = nn.BatchNorm2d(256*block.expansion)
        self.fine_out = nn.Linear(256*block.expansion,num_classes)
        self.coarse_out = nn.Linear(256*block.expansion,coarse_aux[3])
        self.coarse_out2_aux1 = nn.Linear(coarse_aux[3],num_classes)
        # self.softmax = nn.Softmax()

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
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

        aux1 = self.layer1_aux1(x)
        aux1 = self.bn_aux1(aux1)
        aux1 = self.relu(aux1)
        aux1 = self.avgpool(aux1)
        aux1 = aux1.view(aux1.size(0),-1)
        aux1 = self.fc_aux1(aux1)
        merge_aux1 = self.fc2_aux1(aux1)
        merge_aux1 = self.relu(merge_aux1)

        core_path = self.layer1(x)

        aux2 = self.layer2_aux2(core_path)
        aux2 = self.bn_aux2(aux2)
        aux2 = self.relu(aux2)
        aux2 = self.avgpool(aux2)
        aux2 = aux2.view(aux2.size(0),-1)
        aux2 = self.fc_aux2(aux2)
        merge_aux2 = self.fc2_aux2(aux2)
        merge_aux2 = self.relu(merge_aux2)

        core_path = self.layer2(core_path)

        aux3 = self.layer3_aux3(core_path)
        aux3 = self.bn_aux3(aux3)
        aux3 = self.relu(aux3)
        aux3 = self.avgpool(aux3)
        aux3 = aux3.view(aux3.size(0),-1)
        aux3 = self.fc_aux3(aux3)
        merge_aux3 = self.fc2_aux3(aux3)
        merge_aux3 = self.relu(merge_aux3)

        core_path = self.layer3(core_path)

        core_path = self.bn_last(core_path)
        core_path = self.avgpool(core_path)
        core_path = core_path.view(core_path.size(0),-1)
        coarse_out = self.coarse_out(core_path)
        merge_coarse_out = self.coarse_out2_aux1(coarse_out)
        merge_coarse_out = self.relu(merge_coarse_out)

        merge = torch.mul(merge_coarse_out,merge_aux3)
        merge = torch.mul(merge,merge_aux2)
        merge = torch.mul(merge,merge_aux1)
        fine_out = self.fine_out(core_path)
        fine_out = torch.mul(fine_out,merge)
        return fine_out,coarse_out,aux3,aux2,aux1
