import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

import argparse
import json
import subprocess
import os

from network import *
from imsitu  import *

def get_gpu_memory_map():
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ])
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    return gpu_memory

def load_classifier(weights_file, encoder):
    '''Return resnet architecture and get device'''

    # load classifier
    classifier = resnet_modified_small(encoder)
    classifier.load_state_dict(torch.load(weights_file))

    # setup gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    classifier = classifier.to(device)
    if device == 'cuda':
        classifier = torch.nn.DataParallel(classifier)
        cudnn.benchmark = True

    return classifier, device

def __train(train_loader, classifier, discriminator, optimizer, criterion, dev, layer_index):
    classifier.eval()
    discriminator.train()

    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (index, inputs, domains) in enumerate(train_loader):
        domains = domains.squeeze(1)
        inputs, domains = inputs.to(dev), domains.to(dev)
        classes = classifier(inputs)[layer_index]
        print(classes.size())

        optimizer.zero_grad()
        outputs = discriminator(classes)
        loss = criterion(outputs, domains)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += domains.size(0)
        correct += predicted.eq(domains).sum().item()

        stats = 'Training Loss: %.3f | Acc: %.3f%% (%d/%d)' % \
            (train_loss/(batch_idx+1), 100.*correct/total, correct, total)

    return


def __test(test_loader, classifier, discriminator, optimizer, criterion, dev, layer_index, best_acc):
    classifier.eval()
    discriminator.eval()

    test_loss = 0
    correct = 0
    total = 0

    ckpt = 'adv.' + str(layer_index)
    with torch.no_grad():
        for batch_idx, (index, inputs, domains) in enumerate(test_loader):
            domains = domains.squeeze(1)
            inputs, domains = inputs.to(dev), domains.to(dev)
            classes = classifier(inputs)[layer_index]
            print(classes.size())

            outputs = discriminator(classes)
            loss = criterion(outputs, domains)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += domains.size(0)
            correct += predicted.eq(domains).sum().item()

            stats = 'Testing Loss: %.3f | Acc: %.3f%% (%d/%d)' % \
                (test_loss/(batch_idx+1), 100.*correct/total, correct, total)

    acc = 100.*correct/total
    if acc > best_acc:
        # print('Saving..')
        # state = {
        #     'net': discriminator.state_dict(),
        #     'acc': acc,
        # }
        # torch.save(state, './checkpoint/' + ckpt)
        best_acc = acc

    return best_acc

def train_all(args): 
    # load and encode annotations
    train_set = json.load(open(args.train_json))
    test_set = json.load(open(args.test_json))
    encoder = torch.load(args.encoding_file)

    # load classifier
    classifier, device = load_classifier(args.weights_file, encoder)

    # load datasets
    dataset_train = imSituSituation(args.image_dir, train_set, encoder, classifier.train_preprocess(), gender_cls=True)
    print("Train Set Size: {}".format(len(dataset_train)))
    dataset_test = imSituSituation(args.image_dir, test_set, encoder, classifier.test_preprocess(), gender_cls=True)
    print("Test Set Size: {}".format(len(dataset_test)))

    batch_size = 64
    train_loader  = torch.utils.data.DataLoader(dataset_train, batch_size = batch_size, shuffle = True, num_workers = 3) 
    test_loader  = torch.utils.data.DataLoader(dataset_test, batch_size = batch_size, shuffle = False, num_workers = 3)

    '''
    Top-level layer shapes
    (3, 224, 224)
    (64, 56, 56)
    (64, 56, 56)
    (128, 28, 28)
    (256, 14, 14)
    (512, 7, 7)
    (1024)
    (175)
    '''

    # setup discriminator network for each layer
    adv0 = Adversary(3 * 224 * 224)
    adv1 = Adversary(64 * 56 * 56)
    adv2 = Adversary(64 * 56 * 56)
    adv3 = Adversary(128 * 28 * 28)
    adv4 = Adversary(256 * 14 * 14)
    adv5 = Adversary(512 * 7 * 7)
    adv6 = Adversary(1024)
    adv7 = Adversary(175)
    advs = [adv0, adv1, adv2, adv3, adv4, adv5, adv6, adv7]

    # setup optimizers
    optimizers = []
    for i in range(len(advs)):
        advs[i] = advs[i].to(device)
        if device == 'cuda':
            advs[i] = torch.nn.DataParallel(advs[i])
        optimizer = optim.SGD(advs[i].parameters(), lr=0.001, momentum=0.9)
        optimizers.append(optimizer)

    criterion = nn.CrossEntropyLoss()

    best_accs = [0] * len(advs)

    # train all discriminators concurrently
    for epoch in range(args.training_epochs):
        print("="*10)
        print("Epoch {}".format(epoch))
        print("="*10)
        for i in range(len(advs)):
            print("Discriminator {}:".format(i))
            __train(train_loader, classifier, advs[i], optimizers[i], criterion, device, i)
            best_accs[i] = __test(test_loader, classifier, advs[i], optimizers[i], criterion, device, i, best_accs[i])
            print("-"*10)

    print(best_accs)
    return

# Sample execution: CUDA_VISIBLE_DEVICES=0 python discriminator.py models/best.th.tar model_output/encoder data/genders_train.json data/genders_test.json
if __name__ == "__main__":
    parser = argparse.ArgumentParser('Train discriminator on network intermediate layers')
    parser.add_argument("weights_file") 
    parser.add_argument("encoding_file") 
    parser.add_argument("train_json") 
    parser.add_argument("test_json") 
    parser.add_argument("--image_dir", default="./resized_256")
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', default=False, help='resume from checkpoint')
    parser.add_argument("--training_epochs", default=10, help="total number of training epochs", type=int)
    args = parser.parse_args()

    gpu_memory = get_gpu_memory_map()
    print(gpu_memory)
    freest_gpu = min(xrange(len(gpu_memory)), key=gpu_memory.__getitem__)
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(freest_gpu)

    train_all(args)
