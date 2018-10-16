import argparse
import copy
import json
# import matplotlib.pyplot as plt
import time

import torch
from torch import optim

from imsitu import *
import network

# TODO: plot option
# TODO: bring back cuda; use_gpu
device_array = []

TRAIN = "train"
VAL = "val"
PHASES = [TRAIN, VAL]

def train_model(max_epoch, batch_size, eval_frequency, dataloaders, model, optimizer, save_dir, timing=True): 
    time_all = time.time()

    # pmodel = torch.nn.DataParallel(model, device_ids=device_array)
    print_freq = 1
    epoch_steps = 0

    # plot statistics
    train_acc = []
    train_loss = []
    val_acc = []
    val_loss = []
    epochs =[]

    # best model info
    best_acc = 0.0
    best_model = copy.deepcopy(model.state_dict())
  
    for epoch in range(0, max_epoch):  
        print('Epoch {}/{}'.format(epoch, max_epoch - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in PHASES:
            model.train((phase==TRAIN))

            epoch_steps = 0
            running_loss = 0.0
            running_corrects = 0
            train_loss_total = 0

            for i, (index, input, target) in enumerate(dataloaders[phase]):
                epoch_steps += 1
           
                t0 = time.time()
                t1 = time.time()

                # setup inputs
                input_var = torch.autograd.Variable(input)
                target_var = torch.autograd.Variable(target.squeeze(1))
                # input_var = torch.autograd.Variable(input.cuda())
                # target_var = torch.autograd.Variable(target.cuda())
                optimizer.zero_grad()

                # forward pass  
                cls_scores = model(input)
                _, preds = torch.max(cls_scores.data, 1)
                loss = model.loss()(cls_scores, target_var)
                if timing : print "forward time = {}".format(time.time() - t1)
                
                # backpropagate during train time
                if phase == TRAIN:                    
                    t1 = time.time()
                    loss.backward()        
                    optimizer.step()
                    if timing: print "backward time = {}".format(time.time() - t1)
                        
                # update epoch statistics
                running_loss += loss.item()
                running_corrects += torch.sum(preds == target_var.data)
                
                # print stats for training
                if epoch_steps % print_freq == 0:
                    avg_loss = running_loss / (epoch_steps)
                    batch_time = (time.time() - time_all)/ (epoch_steps)
                    print "{} phase: {},{} loss = {:.2f}, avg loss = {:.2f}, batch time = {:.2f}".format(phase, epoch_steps-1, epoch, loss.item(), avg_loss, batch_time)
                    print('-' * 10)

            # print epoch stats
            epoch_loss = running_loss / epoch_steps
            epoch_acc = running_corrects / epoch_steps
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # prepare plots
            if phase == TRAIN:
                train_acc.append(epoch_acc)
                train_loss.append(epoch_loss)
                epochs.append(epoch)
            else:
                val_acc.append(epoch_acc)
                val_loss.append(epoch_loss)

        # Plot Losses After Every Epoch
        # plt.plot(epochs, train_loss, '-o', epochs, val_loss, '-o')
        # plt.title('Loss')
        # plt.show()
        # print()

        # deep copy the best model
        if phase == VAL and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    # Plot Summary
    time_elapsed = time.time() - time_all
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    
    # Plot Accuracy at the end of training
    # plt.plot(epochs, train_acc, '-o', epochs, val_acc, '-o')
    # plt.title('Accuracy')
    # plt.ylim(0.4,1.0)
    # plt.show()

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def train(args):
    # load and encode annotations
    train_set = json.load(open(args.train_json))
    dev_set = json.load(open(args.dev_json))
    encoder = imSituVerbRoleLocalNounEncoder(train_set)
    torch.save(encoder, "data/encoder")

    # load model
    model = network.resnet_modified_small(encoder)
    if args.weights_file is not None:
        model.load_state_dict(torch.load(args.weights_file))

    # load datasets; # TODO: target should be single index
    dataset_train = imSituSituation(args.image_dir, train_set, encoder, model.train_preprocess())
    dataset_dev = imSituSituation(args.image_dir, dev_set, encoder, model.dev_preprocess())

    # setup gpus
    ngpus = 1
    device_array = [i for i in range(0,ngpus)]
    batch_size = args.batch_size * ngpus  

    train_loader  = torch.utils.data.DataLoader(dataset_train, batch_size = batch_size, shuffle = True, num_workers = 3) 
    dev_loader  = torch.utils.data.DataLoader(dataset_dev, batch_size = batch_size, shuffle = True, num_workers = 3) 
    dataloaders = {TRAIN: train_loader, VAL: dev_loader}

    # Adam optimization algorithm: Adaptive per parameter learning rate based on first and second gradient moments
    # Good for problems with sparse gradients (NLP and CV)
    # model.cuda()
    optimizer = optim.Adam(model.parameters(), lr = args.learning_rate , weight_decay = args.weight_decay)
    train_model(args.training_epochs, args.batch_size, args.eval_frequency, dataloaders, model, optimizer, args.output_dir)  

# Sample execution: python train.py data/genders_train.json data/genders_dev.json model_output
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train action recognition network.") 
    parser.add_argument("train_json") 
    parser.add_argument("dev_json")    
    parser.add_argument("output_dir", help="location to put output, such as models, features, predictions")
    parser.add_argument("--image_dir", default="./resized_256", help="location of images to process")
    parser.add_argument("--weights_file", help="the model to start from")
    parser.add_argument("--batch_size", default=64, help="batch size for training", type=int)
    parser.add_argument("--learning_rate", default=1e-5, help="learning rate for ADAM", type=float)
    parser.add_argument("--weight_decay", default=5e-4, help="learning rate decay for ADAM", type=float)  
    parser.add_argument("--eval_frequency", default=500, help="evaluate on dev set every N training steps", type=int) 
    parser.add_argument("--training_epochs", default=20, help="total number of training epochs", type=int)
    args = parser.parse_args()

    train(args)
