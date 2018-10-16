import math
import torch
import torch.nn as nn
import torchvision as tv

def initLinear(linear, val = None):
    if val is None:
        fan = linear.in_features + linear.out_features 
        spread = math.sqrt(2.0) * math.sqrt( 2.0 / fan )
    else:
        spread = val
    linear.weight.data.uniform_(-spread,spread)
    linear.bias.data.uniform_(-spread,spread)

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
        self.n_verbs = encoding.n_verbs()

        # define layers
        self.linear = nn.Linear(7 * 7 * self.base_size(), self.rep_size())
        self.cls = nn.Linear(self.rep_size(), self.n_verbs)
        self.dropout2d = nn.Dropout2d(.5)
        self.dropout = nn.Dropout(.5)
        self.relu = nn.LeakyReLU()
        initLinear(self.linear)

    def train_preprocess(self): return self.train_transform
    def dev_preprocess(self): return self.dev_transform

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
     
        x = self.dropout2d(x)

        features = self.dropout(self.relu(self.linear(x.view(-1, 7*7*self.base_size()))))
        return self.cls(features)

    def loss(self):
        return torch.nn.CrossEntropyLoss()
        