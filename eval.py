import argparse
import json
import os
import time

import torch

from lib.imsitu import *
from lib import imsitu_utils, network, plots

use_gpu = torch.cuda.is_available()
args = []

def softmax(outputs):
    for i, output in enumerate(outputs):
        outputs[i] = torch.nn.Softmax(dim=-1)(output)
    return outputs

def compute_train_distribution(train_data, encoder):
    weights = [0 for i in range(encoder.n_classes())]
    for image_name in train_data:
        image = train_data[image_name]
        verb = image["verb"]
        agents = imsitu_utils.get_agents(image)
        assert(len(agents) == 1)
        class_id = encoder.encode_verb_noun(verb, agents[0])
        weights[class_id] += 1

    return weights

def weigh_scores(scores, weights, encoder):    
    weighted_scores = []
    for sample_scores in scores:
        sample_weighted_scores = [s / w for s,w in zip(sample_scores, weights)]
        weighted_scores.append(sample_weighted_scores)
    return torch.FloatTensor(weighted_scores)

def aggregate_scores_per_verb(weighted_scores, encoder):
    aggregated_weighted_scores = []
    for j, sample_scores in enumerate(weighted_scores):
        aggregate_sample_scores = []
        for i, _ in enumerate(sample_scores):
            ids = encoder.get_gender_ids_for_verb(i)
            verb_score = sum([sample_scores[_id] for _id in ids])
            aggregate_sample_scores.append(verb_score)
        aggregated_weighted_scores.append(aggregate_sample_scores)
    return torch.FloatTensor(aggregated_weighted_scores)

def get_activity_label(labels, encoder):
    return torch.Tensor([encoder.get_verb_id(label) for label in labels.cpu().numpy()])

def evaluate_model(dataloader, model, encoder, weights=None): 
    time_all = time.time()
    model.eval()

    num_samples = 0.0
    running_corrects = 0
    mx = len(dataloader)

    correct_per_class = [0 for i in range(encoder.n_classes())]
    count_per_class = [0 for i in range(encoder.n_classes())]

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
        cls_scores = softmax(model(input_var)[-1])       
            
        # predict
        if args.two_n:
            assert(not (args.domain_fusion and args.no_domain))
            if args.prior_shift: 
                cls_scores = weigh_scores(cls_scores.data, weights, encoder)

            if args.no_domain:
                _, activity_gender_preds = torch.max(cls_scores, 1)
            elif args.domain_fusion:
                cls_scores = aggregate_scores_per_verb(cls_scores, encoder)
                _, activity_gender_preds = torch.max(cls_scores, 1) 
            else:
                exit("Prior shift requires either no_domain or domain_fusion inference.")

            # collect activity labels and predictions
            preds = get_activity_label(activity_gender_preds, encoder)
            targets = get_activity_label(target_var.data, encoder) 
            verb_agent_labels = target_var.data
           
        else:
            # standard inference
            assert(not (args.domain_fusion or args.no_domain or args.prior_shift))
            _, preds = torch.max(cls_scores.data, 1) 
            targets = target_var.data.cpu().numpy()[0]

            # collect (verb, agent) labels
            verb_agent_labels = []
            for annot in target_var.data:
                assert(len(annot)==2)
                verb = annot[0]; agent = annot[1]
                label = encoder.encode_verb_noun(encoder.decode_verb(verb), encoder.decode_noun(agent))
                verb_agent_labels.append(label)

        # update per class metrics
        for label, pred, orig_class in zip(targets, preds, verb_agent_labels):
            count_per_class[orig_class] = count_per_class[orig_class] + 1
            correct_per_class[orig_class] = correct_per_class[orig_class] + (label == pred).item()  

        # update aggregate metrics 
        running_corrects += torch.sum(preds == targets).item()

    # Output Summary
    time_elapsed = time.time() - time_all
    print('Evaluation completed in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    accuracy = running_corrects / num_samples
    print('Accuracy: {:4f}'.format(accuracy))

    # compute mean value accuracy
    mean_per_class_acc = 0
    for verb in encoder.verbs:
        mean_gender_acc = 0
        scale = len(encoder.genders)
        for gender in encoder.genders:
            label = encoder.encode_verb_noun(verb, gender)
            if count_per_class[label] > 0:                    
                mean_gender_acc += correct_per_class[label] / float(count_per_class[label])
            else:
                print(encoder.decode(label))
                scale = scale - 1
        if scale > 0: mean_per_class_acc += mean_gender_acc / scale
    mean_per_class_acc = mean_per_class_acc / len(encoder.verbs)
    print('Mean per Class Accuracy: {:4f}'.format(mean_per_class_acc))

    # # plot accuracy per activity
    # accuracies_per_class = []
    # for correct, count in zip(correct_per_class, count_per_class):
    #     if count > 0: accuracies_per_class.append(correct / float(count))
    #     else: accuracies_per_class.append(-1)
    # plot_name = args.test_type + "_acc_per_class.png"
    # plots.plot_accuracy_per_class(accuracies_per_class, encoder, plot_name)
    
def evaluate():
    # load model
    encoder = torch.load(os.path.join("encoders", args.model))
    print("Number of Train Classes: {}".format(encoder.n_classes()))
    model = network.load_classifier(args.model, encoder, use_gpu)

    # load dataset
    test_set = json.load(open(os.path.join("data", args.test_type+"_test.json")))
    dataset_test = imSituSituation(args.image_dir, test_set, encoder, model.test_preprocess(), test=(not args.two_n))
    print("Test Set Size: {}".format(len(dataset_test)))
    test_loader  = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=3)

    if args.prior_shift:
        train_set = json.load(open(os.path.join("data", "activity_balanced_train.json")))
        weights = compute_train_distribution(train_set, encoder)
        evaluate_model(test_loader, model, encoder, weights) 
    else:
        evaluate_model(test_loader, model, encoder)  



## Baseline Standard Inference ##

# CUDA_VISIBLE_DEVICES=1 python eval.py activity_balanced --model activity_balanced_baseline
# CUDA_VISIBLE_DEVICES=1 python eval.py activity_balanced_men --model activity_balanced_baseline
# CUDA_VISIBLE_DEVICES=1 python eval.py activity_balanced_women --model activity_balanced_baseline

## 2n Inference ##

# CUDA_VISIBLE_DEVICES=1 python eval.py activity_balanced --two_n --no_domain --model activity_balanced_2n
# CUDA_VISIBLE_DEVICES=1 python eval.py activity_balanced_men --two_n --no_domain --model activity_balanced_2n
# CUDA_VISIBLE_DEVICES=1 python eval.py activity_balanced_women --two_n --no_domain --model activity_balanced_2n

# CUDA_VISIBLE_DEVICES=1 python eval.py activity_balanced --two_n --domain_fusion --model activity_balanced_2n
# CUDA_VISIBLE_DEVICES=1 python eval.py activity_balanced_men --two_n --domain_fusion --model activity_balanced_2n
# CUDA_VISIBLE_DEVICES=1 python eval.py activity_balanced_women --two_n --domain_fusion --model activity_balanced_2n

## 2n Prior Shift Inference ##

# CUDA_VISIBLE_DEVICES=1 python eval.py activity_balanced --two_n --no_domain --prior_shift --model activity_balanced_2n
# CUDA_VISIBLE_DEVICES=1 python eval.py activity_balanced_men --two_n --no_domain --prior_shift --model activity_balanced_2n
# CUDA_VISIBLE_DEVICES=1 python eval.py activity_balanced_women --two_n --no_domain --prior_shift --model activity_balanced_2n

# CUDA_VISIBLE_DEVICES=1 python eval.py activity_balanced --two_n --domain_fusion --prior_shift --model activity_balanced_2n
# CUDA_VISIBLE_DEVICES=1 python eval.py activity_balanced_men --two_n --domain_fusion --prior_shift --model activity_balanced_2n
# CUDA_VISIBLE_DEVICES=1 python eval.py activity_balanced_women --two_n --domain_fusion --prior_shift --model activity_balanced_2n

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test action recognition network.") 
    parser.add_argument("test_type") 
    parser.add_argument("--image_dir", default="./resized_256", help="location of images to process")
    parser.add_argument("--model", help="the model to start from")
    parser.add_argument("--batch_size", default=64, help="batch size for training", type=int)
    parser.add_argument("--two_n", action='store_true', default=False, help="set for model trained on 2n classification")
    parser.add_argument("--no_domain", action='store_true', default=False, help="set to True to evaluate with no domain")
    parser.add_argument("--domain_fusion", action='store_true', default=False, help="set to True to evaluate with domain_fusion")   
    parser.add_argument("--prior_shift", action='store_true', default=False, help="set to True to evaluate with prior shift")
    args = parser.parse_args()        

    evaluate()
