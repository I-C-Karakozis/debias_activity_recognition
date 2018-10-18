import argparse
import copy
import json
import time

import torch
from torch import optim

from imsitu import *
import network

use_gpu = torch.cuda.is_available()
device_array = []
args = []

TRAIN = "train"
VAL = "val"
PHASES = [TRAIN, VAL]

def save_checkpoint(state, is_best=True, filename='models/checkpoint.pth.tar'):
    if is_best:
        torch.save(state, 'models/model_best.pth.tar')
    else:
        torch.save(state, filename)

def train_model(max_epoch, batch_size, dataloaders, model, optimizer, save_dir): 
    time_all = time.time()

    # pmodel = torch.nn.DataParallel(model, device_ids=device_array)
    print_freq = 10

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

            epoch_steps = 0.0
            num_samples = 0.0
            running_loss = 0.0
            running_corrects = 0
            train_loss_total = 0

            for i, (index, input, target) in enumerate(dataloaders[phase]): 
                epoch_steps += 1.0
                num_samples += target.size()[0]

                t0 = time.time()
                t1 = time.time()

                # setup inputs
                target = target.squeeze(1)
                if use_gpu:
                    input_var = torch.autograd.Variable(input.cuda())
                    target_var = torch.autograd.Variable(target.cuda())
                else:
                    input_var = torch.autograd.Variable(input)
                    target_var = torch.autograd.Variable(target)                    
                optimizer.zero_grad()

                # forward pass  
                cls_scores = model(input_var)
                _, preds = torch.max(cls_scores.data, 1)
                loss = model.loss()(cls_scores, target_var)
                if args.timing : print "forward time = {}".format(time.time() - t1)
                
                # backpropagate during train time
                if phase == TRAIN:                    
                    t1 = time.time()
                    loss.backward()        
                    optimizer.step()
                    if args.timing: print "backward time = {}".format(time.time() - t1)
                        
                # update epoch statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == target_var.data)
                
                # print stats for training
                if epoch_steps % print_freq == 0:
                    avg_loss = running_loss / (epoch_steps)
                    batch_time = (time.time() - time_all)/ (epoch_steps)
                    print "{} phase: {},{} loss = {:.2f}, avg loss = {:.2f}, batch time = {:.2f}".format(phase, epoch_steps-1, epoch, loss.data[0], avg_loss, batch_time)
                    print('-' * 10)

            # print epoch stats
            epoch_loss = running_loss / epoch_steps
            epoch_acc = running_corrects / num_samples
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
        if args.plot:
            plt.plot(epochs, train_loss, '-o', epochs, val_loss, '-o')
            plt.title('Loss')
            plt.savefig("figures/loss")

        # deep copy the best model
        if phase == VAL and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model = copy.deepcopy(model.state_dict())

    # Plot Summary
    time_elapsed = time.time() - time_all
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    
    # Plot Accuracy at the end of training
    if args.plot:
        plt.plot(epochs, train_acc, '-o', epochs, val_acc, '-o')
        plt.title('Accuracy')
        plt.savefig("figures/accuracy")

    # load best model weights
    save_checkpoint(model, is_best=False, filename="models/final.pth.tar")
    model.load_state_dict(best_model)
    save_checkpoint(model)

def train():
    # load and encode annotations
    train_set = json.load(open(args.train_json))
    dev_set = json.load(open(args.dev_json))
    encoder = imSituVerbRoleLocalNounEncoder(train_set)
    torch.save(encoder, "data/encoder")

    # load model
    model = network.resnet_modified_small(encoder)
    if args.weights_file is not None:
        model.load_state_dict(torch.load(args.weights_file))

    # load datasets
    dataset_train = imSituSituation(args.image_dir, train_set, encoder, model.train_preprocess())
    print("Train Set Size: {}".format(len(dataset_train)))
    dataset_dev = imSituSituation(args.image_dir, dev_set, encoder, model.dev_preprocess())
    print("Validation Set Size: {}".format(len(dataset_dev)))
    
    # setup gpus
    ngpus = 1
    device_array = [i for i in range(0,ngpus)]
    batch_size = args.batch_size * ngpus  

    train_loader  = torch.utils.data.DataLoader(dataset_train, batch_size = batch_size, shuffle = True, num_workers = 3) 
    dev_loader  = torch.utils.data.DataLoader(dataset_dev, batch_size = batch_size, shuffle = True, num_workers = 3) 
    dataloaders = {TRAIN: train_loader, VAL: dev_loader}

    # Adam optimization algorithm: Adaptive per parameter learning rate based on first and second gradient moments
    # Good for problems with sparse gradients (NLP and CV)
    if use_gpu: model.cuda()
    optimizer = optim.Adam(model.parameters(), lr = args.learning_rate , weight_decay = args.weight_decay)
    train_model(args.training_epochs, args.batch_size, dataloaders, model, optimizer, args.output_dir)  

# Sample execution: CUDA_VISIBLE_DEVICES=3 python train.py data/genders_train.json data/genders_dev.json model_output --plot > model_output/logs
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
    parser.add_argument("--training_epochs", default=20, help="total number of training epochs", type=int)
    parser.add_argument("--plot", action='store_true', default=False, help="set to True to produce plots")
    parser.add_argument("--timing", action='store_true', default=False, help="set to True to time each pass through the network")
    args = parser.parse_args()

    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.ioff()

    train()
