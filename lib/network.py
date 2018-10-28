import math
import os
import torch
import torch.nn as nn
import torchvision as tv

class Adversary(nn.Module):
    def __init__(self, insize):
        super(Adversary, self).__init__()
        num_domains = 2
        self.insize = insize
        self.linear1 = nn.Linear(insize, num_domains)

    def forward(self, x):
       x = x.view(-1, self.insize)
       return self.linear1(x)

    def loss(self):
        return torch.nn.CrossEntropyLoss()

def initLinear(linear, val = None):
    if val is None:
        fan = linear.in_features + linear.out_features 
        spread = math.sqrt(2.0) * math.sqrt( 2.0 / fan )
    else:
        spread = val
    linear.weight.data.uniform_(-spread,spread)
    linear.bias.data.uniform_(-spread,spread)

def load_classifier(weights_file, encoder, use_gpu):
    '''Return resnet architecture'''
    classifier = resnet_modified_small(encoder)
    if weights_file is not None:
        model_name = os.path.join("models", weights_file + ".pth.tar") 
        classifier.load_state_dict(torch.load(model_name))
    if use_gpu: classifier.cuda()

    return classifier

class resnet_modified_small(nn.Module):
    def base_size(self): return 512
    def rep_size(self): return 1024

    def __init__(self, encoding):
        super(resnet_modified_small, self).__init__()
        self.resnet = tv.models.resnet34(pretrained=True)

        # setup dataset preprocessing
        self.normalize = tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.train_transform = tv.transforms.Compose([
            tv.transforms.Resize(224),
            tv.transforms.RandomCrop(224),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            self.normalize,
        ])
        self.dev_transform = tv.transforms.Compose([
            tv.transforms.Resize(224),
            tv.transforms.CenterCrop(224),
            tv.transforms.ToTensor(),
            self.normalize,
        ])

        # setup encoder
        self.encoding = encoding
        self.n_classes = encoding.n_classes()

        # define layers
        self.linear = nn.Linear(7 * 7 * self.base_size(), self.rep_size())
        self.cls = nn.Linear(self.rep_size(), self.n_classes)
        self.dropout2d = nn.Dropout2d(.5)
        self.dropout = nn.Dropout(.5)
        self.relu = nn.LeakyReLU()
        initLinear(self.linear)

    def train_preprocess(self): return self.train_transform
    def dev_preprocess(self): return self.dev_transform
    def test_preprocess(self): return self.dev_transform

    def forward(self, out0):
        x = self.resnet.conv1(out0)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        out1 = self.resnet.maxpool(x)

        out2 = self.resnet.layer1(out1)
        out3 = self.resnet.layer2(out2)
        out4 = self.resnet.layer3(out3)
        out5 = self.resnet.layer4(out4)
     
        x = self.dropout2d(out5)

        features = self.dropout(self.relu(self.linear(x.view(-1, 7*7*self.base_size()))))
        cls_scores = self.cls(features)
        return [out0, out1, out2, out3, out4, out5, features, cls_scores]

    def loss(self):
        return torch.nn.CrossEntropyLoss()
        