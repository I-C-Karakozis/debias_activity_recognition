import argparse
import json
import time

import torch

from lib.imsitu import *
from lib import network, plots

use_gpu = torch.cuda.is_available()

def compute_weights(scores, encoder):
    weighted_scores = []
    for sample_scores in scores:
        sample_weighted_scores = []
        for i, score in enumerate(sample_scores):
            ids = encoder.get_activity_ids(i)
            weight = sum([sample_scores[_id] for _id in ids])
            sample_weighted_scores.append(score / weight)
        weighted_scores.append(sample_weighted_scores)
    return torch.FloatTensor(weighted_scores)

def evaluate_model(dataloader, model, encoder, plot_name): 
    time_all = time.time()
    model.eval()

    num_samples = 0.0
    running_corrects = 0
    mx = len(dataloader)

    print(encoder.n_classes())
    correct_per_activity = [0 for i in range(encoder.n_classes())]
    count_per_activity = [0 for i in range(encoder.n_classes())]

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
        if args.prior_shift:
            weighted_scores = compute_weights(cls_scores.data, encoder)
            _, preds = torch.max(weighted_scores, 1)
        else:         
            _, preds = torch.max(cls_scores.data, 1)       
        running_corrects += torch.sum(preds == target_var.data)

        # update per class metrics
        for label, pred in zip(target_var.data, preds):
            count_per_activity[label] = count_per_activity[label] + 1
            correct_per_activity[label] = correct_per_activity[label] + (label == pred)

    # Output Summary
    time_elapsed = time.time() - time_all
    print('Evaluation completed in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    accuracy = running_corrects / num_samples
    print('Accuracy: {:4f}'.format(accuracy))

    if args.prior_shift:
        mean_verb_acc = 0
        for verb in encoder.verbs:
            mean_gender_acc = 0
            for gender in encoder.genders:
                label = encoder.encode_verb_noun(verb, gender)
                if count_per_activity[label] > 0:                    
                    mean_gender_acc += correct_per_activity[label] / float(count_per_activity[label])
                else:
                    print(encoder.decode(label))
            mean_verb_acc += mean_gender_acc / len(encoder.genders)
        mean_verb_acc = mean_verb_acc / len(encoder.verbs)

    else:
        # plot accuracy per activity
        accuracies_per_activity = []
        for correct, count in zip(correct_per_activity, count_per_activity):
            if count > 0: accuracies_per_activity.append(correct / float(count))
            else: accuracies_per_activity.append(-1)
        plots.plot_accuracy_per_activity(accuracies_per_activity, encoder, plot_name)
    
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

    evaluate_model(test_loader, model, encoder, args.plot_name)  

# Sample execution:

# CUDA_VISIBLE_DEVICES=1 python eval.py data/activity_balanced_test.json encoders/activity_balanced_encoder acc_per_activity_per_gender.png --prior_shift --weights_file models/activity_balanced_best.pth.tar

# CUDA_VISIBLE_DEVICES=1 python eval.py data/balanced_genders_test.json encoders/encoder balanced_model_balanced_test_acc_per_activity.png --weights_file models/best.pth.tar
# CUDA_VISIBLE_DEVICES=1 python eval.py data/skewed_genders_test.json encoders/encoder balanced_model_skewed_test_acc_per_activity.png --weights_file models/best.pth.tar

# CUDA_VISIBLE_DEVICES=1 python eval.py data/balanced_genders_test.json encoders/encoder skewed_model_balanced_test_acc_per_activity.png --weights_file models/skewed_best.pth.tar
# CUDA_VISIBLE_DEVICES=1 python eval.py data/skewed_genders_test.json encoders/encoder skewed_model_skewed_test_acc_per_activity.png --weights_file models/skewed_best.pth.tar

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test action recognition network.") 
    parser.add_argument("test_json") 
    parser.add_argument("encoding_file") 
    parser.add_argument("plot_name", help="Name of plot file; it will be stored automatically in figures/plot_name") 
    parser.add_argument("--image_dir", default="./resized_256", help="location of images to process")
    parser.add_argument("--weights_file", help="the model to start from")
    parser.add_argument("--batch_size", default=64, help="batch size for training", type=int)
    parser.add_argument("--prior_shift", action='store_true', default=False, help="set to True to evaluate with prior prior_shift")
    args = parser.parse_args()        

    evaluate()

# Skewed Test Set Size: 5886
# Orig Accuracy: 0.341148

# Balanced Test Set Size: 2220
# Orig Accuracy: 0.310360
# Balanced Accuracy: 0.354505
# Skewed Accuracy: 0.346847

# Skewed Test Set Size: 3005
# Balanced Accuracy: 0.358735
# Skewed Accuracy: 0.368386
