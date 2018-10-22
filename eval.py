import argparse
import json
import time

import torch

from lib.imsitu import *
from lib import network, plots

use_gpu = torch.cuda.is_available()

def evaluate_model(dataloader, model, encoder): 
    time_all = time.time()
    model.eval()

    num_samples = 0.0
    running_corrects = 0
    mx = len(dataloader)

    correct_per_activity = [0] * encoder.n_verbs
    count_per_activity = [0] * encoder.n_verbs

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
        cls_scores = model(input_var)[-1]
        _, preds = torch.max(cls_scores.data, 1)                
        running_corrects += torch.sum(preds == target_var.data)

        # update per activity metrics
        for label, pred in zip(target_var.data, cls_scores):
            count_per_activity[label] = count_per_activity[label] + 1
            correct_per_activity[label] = correct_per_activity[label] + (label == pred)

    # Output Summary
    time_elapsed = time.time() - time_all
    print('Evaluation completed in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    accuracy = running_corrects / num_samples
    print('Accuracy: {:4f}'.format(accuracy))

    # plot accuracy per activity
    accuracies_per_activity = [correct / float(count) for correct, count in zip(correct_per_activity, count_per_activity) if count > 0 else -1]
    plots.plot_accuracy_per_activity(accuracies_per_activity, encoder)
    
def evaluate():
    # load annotations
    encoder = torch.load(args.encoding_file)
    test_set = json.load(open(args.test_json))

    # load model
    model = network.load_classifier(args.weights_file, encoder, use_gpu)

    # load dataset
    dataset_test = imSituSituation(args.image_dir, test_set, encoder, model.test_preprocess())
    print("Test Set Size: {}".format(len(dataset_test)))
    batch_size = args.batch_size 
    test_loader  = torch.utils.data.DataLoader(dataset_test, batch_size = batch_size, shuffle = False, num_workers = 3)

    evaluate_model(test_loader, model, encoder)  

# Sample execution: 
# CUDA_VISIBLE_DEVICES=0 python eval.py data/genders_test.json model_output/encoder --weights_file models/best.pth.tar
# CUDA_VISIBLE_DEVICES=1 python eval.py data/balanced_genders_test.json model_output/encoder --weights_file models/best.pth.tar
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test action recognition network.") 
    parser.add_argument("test_json") 
    parser.add_argument("encoding_file") 
    parser.add_argument("--image_dir", default="./resized_256", help="location of images to process")
    parser.add_argument("--weights_file", help="the model to start from")
    parser.add_argument("--batch_size", default=64, help="batch size for training", type=int)
    args = parser.parse_args()        

    evaluate()

# Skewed Test Set Size: 5886
# Accuracy: 0.341148

# Balanced Test Set Size: 2220
# Accuracy: 0.310360
