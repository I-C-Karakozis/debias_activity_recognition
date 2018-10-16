import argparse
import json
import time

import torch
from torch import optim

from imsitu import *
import network

# TODO: bring back cuda
device_array = []

def train_model(max_epoch, eval_frequency, train_loader, dev_loader, model, encoding, optimizer, save_dir, timing = False): 
    model.train()

    time_all = time.time()

    pmodel = torch.nn.DataParallel(model, device_ids=device_array)
    top1 = imSituTensorEvaluation(1, 1, encoding)
    top5 = imSituTensorEvaluation(5, 1, encoding)
    loss_total = 0 
    print_freq = 10
    total_steps = 0
    avg_scores = []
  
    for k in range(0,max_epoch):  
      for i, (index, input, target) in enumerate(train_loader):
        total_steps += 1
   
        t0 = time.time()
        t1 = time.time() 
      
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        # input_var = torch.autograd.Variable(input.cuda())
        # target_var = torch.autograd.Variable(target.cuda())
        print(pmodel(input_var).size())
        (_,v,vrn,norm,scores,predictions)  = pmodel(input_var)
        (s_sorted, idx) = torch.sort(scores, 1, True)
        
        # time forward pass
        if timing : print "forward time = {}".format(time.time() - t1)
        optimizer.zero_grad()

        # print loss
        t1 = time.time()
        loss = model.mil_loss(v, vrn, norm, target, 3)
        if timing: print "loss time = {}".format(time.time() - t1)
        
        # time backpropagation
        t1 = time.time()
        loss.backward()        
        if timing: print "backward time = {}".format(time.time() - t1)
        optimizer.step()
        loss_total += loss.data[0]
        return
        
        # score situation
        t2 = time.time() 
        top1.add_point(target, predictions.data, idx.data)
        top5.add_point(target, predictions.data, idx.data)
     
        if timing: print "eval time = {}".format(time.time() - t2)
        if timing: print "batch time = {}".format(time.time() - t0)
        if total_steps % print_freq == 0:
           top1_a = top1.get_average_results()
           top5_a = top5.get_average_results()
           print "{},{},{}, {} , {}, loss = {:.2f}, avg loss = {:.2f}, batch time = {:.2f}".format(total_steps-1,k,i, format_dict(top1_a, "{:.2f}", "1-"), format_dict(top5_a,"{:.2f}","5-"), loss.data[0], loss_total / ((total_steps-1)%eval_frequency) , (time.time() - time_all)/ ((total_steps-1)%eval_frequency))
        if total_steps % eval_frequency == 0:
          print "eval..."    
          etime = time.time()
          (top1, top5) = eval_model(dev_loader, encoding, model)
          model.train() 
          print "... done after {:.2f} s".format(time.time() - etime)
          top1_a = top1.get_average_results()
          top5_a = top5.get_average_results()

          avg_score = top1_a["verb"] + top1_a["value"] + top1_a["value-all"] + top5_a["verb"] + top5_a["value"] + top5_a["value-all"] + top5_a["value*"] + top5_a["value-all*"]
          avg_score /= 8

          print "Dev {} average :{:.2f} {} {}".format(total_steps-1, avg_score*100, format_dict(top1_a,"{:.2f}", "1-"), format_dict(top5_a, "{:.2f}", "5-"))
          
          avg_scores.append(avg_score)
          maxv = max(avg_scores)

          if maxv == avg_scores[-1]: 
            torch.save(model.state_dict(), save_dir + "/{0}.model".format(maxv))   
            print "new best model saved! {0}".format(maxv)

          top1 = imSituTensorEvaluation(1, 3, encoding)
          top5 = imSituTensorEvaluation(5, 3, encoding)
          loss_total = 0
          time_all = time.time()

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

    # Adam optimization algorithm: Adaptive per parameter learning rate based on first and second gradient moments
    # Good for problems with sparse gradients (NLP and CV)
    # model.cuda()
    optimizer = optim.Adam(model.parameters(), lr = args.learning_rate , weight_decay = args.weight_decay)
    train_model(args.training_epochs, args.eval_frequency, train_loader, dev_loader, model, encoder, optimizer, args.output_dir)  

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
