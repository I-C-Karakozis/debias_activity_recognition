import argparse
import copy
import json
import os
import sys

from lib.imsitu_utils import *

# threshold used in the paper: 85 (resulting in 212 verbs)
# if we remove images containing both men and women, we get 203 verbs
THRESHOLD = 85
TWON_THRESHOLD = 40
MIN_INSTANCES_PER_GENDER = 20

SKEW_NUM = 3
SKEW_DENOM = 4
UPPERBOUND_CHECK = 200

def collect_dataset(data, man, woman, human_verbs, human_count, output_prefix):
    # write adjusted dataset
    output = dict()
    sys.stdout = open(os.path.join("stats", output_prefix+"_"+args.data_json.split("/")[-1].split(".")[0]+".txt"), "w")
    print_stats(human_count, human_verbs)
    for image_name in data:
        collect_limited_human_activities(image_name, human_count, human_verbs, man, woman, data, output)

    with open(os.path.join("data", output_prefix + "_train.json"), "w") as f:     
        json.dump(output, f) 

    # store verbs of humans in action
    with open(os.path.join("verbs", output_prefix + ".txt"), 'w') as f:
        for verb in human_verbs:
            f.write("%s\n" % verb)

def prepare_activity_balanced_dataset(data, man, woman, human_count, output_prefix):
    # find number of images per verb that maximizes number of train images
    max_per_verb_image_count = 0; max_total_image_count = 0
    for i in range(TWON_THRESHOLD, UPPERBOUND_CHECK):
        # enforce same number of images per verb
        human_verbs = [k for k in human_count if sum(human_count[k]) >= i and min(human_count[k]) >= MIN_INSTANCES_PER_GENDER]
        current = i * len(human_verbs)

        # print("Images per Activity", "Activities", "Total Images")
        # print(i, len(human_verbs), current)
        # print_border()

        if current > max_total_image_count:
            max_total_image_count = current
            max_per_verb_image_count = i
    # print(max_total_image_count, max_per_verb_image_count)
    human_verbs = [k for k in human_count if sum(human_count[k]) >= max_per_verb_image_count and min(human_count[k]) >= MIN_INSTANCES_PER_GENDER]

    # measure skew per verb
    man_skew = dict()
    for verb in human_verbs:
        man_skew[verb] = human_count[verb][0] / float(sum(human_count[verb]))

    # enforce original skews per verb
    skewed_human_count = dict()
    for verb in human_verbs:
        skewed_human_count[verb] = [0, 0]
        skewed_human_count[verb][0] = round(max_per_verb_image_count * man_skew[verb]) 
        skewed_human_count[verb][1] = max_per_verb_image_count - skewed_human_count[verb][0]
    
    collect_dataset(data, man, woman, human_verbs, skewed_human_count, output_prefix)

def prepare_balanced_and_skewed_datasets(data, man, woman, human_count, output_prefix):
    for verb in human_count:
        # identify skew direction
        if human_count[verb][0] > human_count[verb][1]: hi = 0; lo = 1
        else: hi = 1; lo = 0

        # form pre-skew, pre-balance ratios
        while human_count[verb][hi] % 3 != 0: human_count[verb][hi] = human_count[verb][hi] - 1
        while human_count[verb][lo] % 2 != 0: human_count[verb][lo] = human_count[verb][lo] - 1
        while human_count[verb][hi] > 3 * human_count[verb][lo] / 2: human_count[verb][hi] = human_count[verb][hi] - 1
        while human_count[verb][lo] > 2 * human_count[verb][hi] / 3: human_count[verb][lo] = human_count[verb][lo] - 1

    # find number of images per verb that maximizes number of train images
    max_per_verb_image_count = 0; max_total_image_count = 0
    for i in range(THRESHOLD, UPPERBOUND_CHECK, 5):
        # enforce same number of images per verb
        human_verbs = [k for k in human_count if sum(human_count[k]) >= i]
        current = i * len(human_verbs)

        # enforce same number of images across genders
        man_skew_count = 0; woman_skew_count = 0
        for verb in human_verbs:
            if human_count[verb][0] > human_count[verb][1]:
                man_skew_count = man_skew_count + 1
            else: 
                woman_skew_count = woman_skew_count + 1

        # compute resulting dataset statistics
        image_per_verb_used = 4 * i / 5
        num_verbs = 2 * min(man_skew_count, woman_skew_count)
        current = image_per_verb_used * num_verbs

        if current > max_total_image_count:
            max_total_image_count = current
            max_per_verb_image_count = i
    print(max_total_image_count, max_per_verb_image_count)

    # compute skews
    human_verbs = [k for k in human_count if sum(human_count[k]) >= max_per_verb_image_count]
    man_skew_count = 0; woman_skew_count = 0
    for verb in human_verbs:
        if human_count[verb][0] > human_count[verb][1]:
            man_skew_count = man_skew_count + 1
        else: 
            woman_skew_count = woman_skew_count + 1
    man_skew_count = min(man_skew_count, woman_skew_count)
    woman_skew_count = min(man_skew_count, woman_skew_count)

    # prepare dataset sample ratios between genders and activities
    balanced_human_count = dict(); skewed_human_count = dict()
    for verb in human_verbs:
        # identify skew direction
        if human_count[verb][0] > human_count[verb][1]: hi = 0; lo = 1
        else: hi = 1; lo = 0

        # form balanced dataset
        if hi == 0 and man_skew_count > 0:
            man_skew_count = man_skew_count - 1
        elif hi == 1 and woman_skew_count > 0:
            woman_skew_count = woman_skew_count - 1
        else:
            continue

        # form balanced dataset
        balanced_human_count[verb] = [0] * 2
        balanced_human_count[verb][hi] = 2 * max_per_verb_image_count / 5
        balanced_human_count[verb][lo] = 2 * max_per_verb_image_count / 5

        # form skewed dataset
        skewed_human_count[verb] = [0] * 2
        skewed_human_count[verb][hi] = 3 * max_per_verb_image_count / 5
        skewed_human_count[verb][lo] = max_per_verb_image_count / 5
    
    # collect balanced and skewed dataset annotations
    human_verbs = [k for k in balanced_human_count]
    balanced_output_prefix = "balanced_" + output_prefix
    collect_dataset(data, man, woman, human_verbs, balanced_human_count, balanced_output_prefix)
    skewed_output_prefix = "skewed_" + output_prefix
    collect_dataset(data, man, woman, human_verbs, skewed_human_count, skewed_output_prefix)

def parse_json(args):
    # collect gender encodings
    nouns, verbs = get_space()
    man = get_noun(nouns, MAN)
    woman = get_noun(nouns, WOMAN)

    # collect human activities
    data = json.load(open(args.data_json))
    human_count = count_human_activities(data, man, woman)
    
    # collect datasets to form annotations for 
    assert(not (args.balanced_and_skewed and args.activity_balanced))
    if args.balanced_and_skewed:
        prepare_balanced_and_skewed_datasets(data, man, woman, human_count, args.output_prefix)
    elif args.activity_balanced:
        prepare_activity_balanced_dataset(data, man, woman, human_count, args.output_prefix)
    else:
        human_verbs = [k for k in human_count if sum(human_count[k]) > THRESHOLD and min(human_count[k]) >= MIN_INSTANCES_PER_GENDER]
        collect_dataset(data, man, woman, human_verbs, human_count, args.output_prefix)

# Sample execution: 
# python preprocess_train.py imsitu_data/train.json gender
# python preprocess_train.py imsitu_data/train.json fixed_gender_ratio --balanced_and_skewed
# python preprocess_train.py imsitu_data/train.json activity_balanced --activity_balanced
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Fetch all actions with man or woman agents.") 
  parser.add_argument("data_json", help="Input dataset json to preprocess.") 
  parser.add_argument("output_prefix", help="Output dataset json to collect.")
  parser.add_argument("--balanced_and_skewed", action='store_true', default=False, help="set to True to form balanced and skewed datasets")
  parser.add_argument("--activity_balanced", action='store_true', default=False, help="set to True to form dataset with same number of images per activity")
  args = parser.parse_args()

  parse_json(args)
