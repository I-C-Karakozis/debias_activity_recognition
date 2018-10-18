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

# TODO: parallelize
use_gpu = torch.cuda.is_available()

def __train(train_loader, classifier, discriminator, optimizer, layer_index):
    classifier.eval()
    discriminator.train()

    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (index, inputs, domains) in enumerate(train_loader):
        domains = domains.squeeze(1)
        if use_gpu: 
            inputs = torch.autograd.Variable(inputs.cuda())
            domains = torch.autograd.Variable(domains.cuda())
        classes = classifier(inputs)[layer_index]
        print(classes.size())

        optimizer.zero_grad()
        outputs = discriminator(classes)
        loss = classifier.loss()(outputs, domains)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = outputs.max(1)
        total += domains.size(0)
        correct += predicted.eq(domains).sum().data[0]

        stats = 'Training Loss: %.3f | Acc: %.3f%% (%d/%d)' % \
            (train_loss/(batch_idx+1), 100.*correct/total, correct, total)
        print(stats)

    return

def __test(test_loader, classifier, discriminator, optimizer, layer_index, best_acc):
    classifier.eval()
    discriminator.eval()

    test_loss = 0
    correct = 0
    total = 0

    ckpt = 'adv.' + str(layer_index)
    for batch_idx, (index, inputs, domains) in enumerate(test_loader):
        domains = domains.squeeze(1)
        if use_gpu: 
            inputs = torch.autograd.Variable(inputs.cuda())
            domains = torch.autograd.Variable(domains.cuda())
        classes = classifier(inputs)[layer_index]
        print(classes.size())

        outputs = discriminator(classes)
        loss = classifier.loss()(outputs, domains)

        test_loss += loss.data[0]
        _, predicted = outputs.max(1)
        total += domains.size(0)
        correct += predicted.eq(domains).sum().data[0]

        stats = 'Testing Loss: %.3f | Acc: %.3f%% (%d/%d)' % \
            (test_loss/(batch_idx+1), 100.*correct/total, correct, total)
        print(stats)

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
    classifier = load_classifier(args.weights_file, encoder, use_gpu)

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
        if use_gpu: advs[i] = advs[i].cuda()
        optimizer = optim.SGD(advs[i].parameters(), lr=0.001, momentum=0.9)
        optimizers.append(optimizer)

    # train and evaluate all discriminators concurrently
    best_accs = [0] * len(advs)
    for epoch in range(args.training_epochs):
        print("="*10)
        print("Epoch {}".format(epoch))
        print("="*10)
        for i in range(len(advs)):
            print("Discriminator {}:".format(i))
            __train(train_loader, classifier, advs[i], optimizers[i], i)
            best_accs[i] = __test(test_loader, classifier, advs[i], optimizers[i], i, best_accs[i])
            print("-"*10)


    # plot best accuracy per layer
    print(best_accs)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.ioff()
    plt.plot([i for i in range(len(best_accs))], a, '-o')
    plt.savefig("figures/discriminators_gender_cls_acc")

# Sample execution: CUDA_VISIBLE_DEVICES=0 python discriminator.py model_output/encoder data/genders_train.json data/genders_test.json --weights_file models/best.pth.tar
if __name__ == "__main__":
    parser = argparse.ArgumentParser('Train discriminator on network intermediate layers')
    parser.add_argument("encoding_file") 
    parser.add_argument("train_json") 
    parser.add_argument("test_json") 
    parser.add_argument("--weights_file", help="the model to start from")
    parser.add_argument("--image_dir", default="./resized_256")
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', default=False, help='resume from checkpoint')
    parser.add_argument("--training_epochs", default=10, help="total number of training epochs", type=int)
    args = parser.parse_args()

    train_all(args)

# best_acc = [55.589534488617055, 62.249405368671425, 62.11348963642541, 65.83418280665987, 73.30954808019028, 72.08630648997621, 73.07169554875976, 72.579001019368]