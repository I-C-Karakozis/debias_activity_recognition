import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

import argparse
import os

from network import *

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
    classifier = network.resnet_modified_small(encoder)
    classifier.load_state_dict(torch.load(weights_file))

    # setup gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    classifier = classifier.to(device)
    if device == 'cuda':
        classifier = torch.nn.DataParallel(classifier)
        cudnn.benchmark = True

    return classifier, device

def train_all(args): 
    # load and encode annotations
    train_set = json.load(open(args.train_json))
    test_set = json.load(open(args.test_json))
    encoder = args.encoding_file

    # load classifier
    classifier, device = load_classifier(args.weights_file, encoder)
    classifier.eval()

    # load datasets
    dataset_train = imSituSituation(args.image_dir, train_set, encoder, classifier.train_preprocess())
    print("Train Set Size: {}".format(len(dataset_train)))
    dataset_test = imSituSituation(args.image_dir, test_set, encoder, classifier.test_preprocess())
    print("Test Set Size: {}".format(len(dataset_test)))

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
    adv2 = Adversary(128 * 28 * 28)
    adv3 = Adversary(256 * 14 * 14)
    adv4 = Adversary(512 * 7 * 7)
    adv5 = Adversary(1024)
    adv6 = Adversary(175)
    advs = [adv0, adv1, adv2, adv3, adv4, adv5, adv6]

    # setup optimizers
    optimizers = []
    for i in range(len(advs)):
        advs[i] = advs[i].to(device)
        if device == 'cuda':
            advs[i] = torch.nn.DataParallel(advs[i])
        advs[i].train()
        optimizer = optim.SGD(advs[i].parameters(), lr=0.001, momentum=0.9)
        optimizers.append(optimizer)
        
    criterion = nn.CrossEntropyLoss()

    best_accs = [0] * len(advs)

    # train all discriminators concurrently
    for k in range(epochs):
        for i in range(len(advs)):
            __train(trainloader, classifier, advs[i], optimizers[i],
                    criterion, device, i)
            best_accs[i] = __test(testloader, classifier, advs[i],
                    optimizers[i], criterion, device, best_accs[i], i)

    print(best_accs)
    return

def __train(trainloader, classifier, discriminator, optimizer, criterion, dev,
        layer_index):
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        domains = targets[:][1]
        inputs, targets = inputs.to(dev), domains.to(dev)
        classes = classifier(inputs)[layer_index]

        optimizer.zero_grad()
        outputs = discriminator(classes)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        stats = 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % \
            (train_loss/(batch_idx+1), 100.*correct/total, correct, total)
    return


def __test(testloader, classifier, discriminator, optimizer, criterion, dev,
        best_acc, layer_index):
    classifier.eval()
    discriminator.eval()
    test_loss = 0
    correct = 0
    total = 0
    ckpt = 'adv.' + str(layer_index)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
             inputs, targets = inputs.to(dev), domains.to(dev)
            classes = classifier(inputs)[layer_index]

            outputs = discriminator(classes)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            stats = 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % \
                (test_loss/(batch_idx+1), 100.*correct/total, correct, total)


    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': discriminator.state_dict(),
            'acc': acc,
        }
        torch.save(state, './checkpoint/' + ckpt)
        best_acc = acc
    return best_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train discriminator on network intermediate layers')
    parser.add_argument("weights_file") 
    parser.add_argument("encoding_file") 
    parser.add_argument("train_json") 
    parser.add_argument("test_json") 
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
