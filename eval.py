import argparse
import json
import time

import torch

from imsitu import *
import network

use_gpu = torch.cuda.is_available()

def evaluate_model(dataloader, model): 
    time_all = time.time()
    model.eval()

    num_samples = 0.0
    running_corrects = 0
    mx = len(dataloader)

    for i, (indexes, input, target) in enumerate(dataloader):
        if i % 10 == 0: print("batch {} out of {}\r".format(i+1,mx))
        num_samples += target.size()[0]

        # setup inputs
        target = target.squeeze(1)
        if use_gpu:
            input_var = torch.autograd.Variable(input.cuda())
            target_var = torch.autograd.Variable(target.cuda())
        else:
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)        
            
        # evaluate
        cls_scores = model(input_var)
        _, preds = torch.max(cls_scores.data, 1)                
        running_corrects += torch.sum(preds == target_var.data)

    # Plot Summary
    time_elapsed = time.time() - time_all
    print('Evaluation completed in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    accuracy = running_corrects / num_samples
    print('Accuracy: {:4f}'.format(accuracy))
    
def evaluate():
    # load annotations
    encoder = torch.load(args.encoding_file)
    test_set = json.load(open(args.test_json))

    # load model
    model = network.resnet_modified_small(encoder)
    if args.weights_file is not None:
        model.load_state_dict(torch.load(args.weights_file))

    # load dataset
    dataset_test = imSituSituation(args.image_dir, test_set, encoder, model.test_preprocess())
    print("Test Set Size: {}".format(len(dataset_test)))
    batch_size = args.batch_size 
    test_loader  = torch.utils.data.DataLoader(dataset_test, batch_size = batch_size, shuffle = False, num_workers = 3)
    
    if use_gpu: model.cuda()
    evaluate_model(test_loader, model)  

# Sample execution: CUDA_VISIBLE_DEVICES=0 python eval.py data/genders_test.json model_output/encoder --weights_file models/model_best.pth.tar
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test action recognition network.") 
    parser.add_argument("test_json") 
    parser.add_argument("encoding_file") 
    parser.add_argument("--image_dir", default="./resized_256", help="location of images to process")
    parser.add_argument("--weights_file", help="the model to start from")
    parser.add_argument("--batch_size", default=64, help="batch size for training", type=int)
    args = parser.parse_args()        

    evaluate()

