import argparse
import copy
import json
import os

from imsitu_utils import *

# TODO: concatenate train and dev for evaluation

# threshold used in the paper: 85 (resulting in 212 verbs)
# if we remove images containing both men and women, we get 203 verbs
THRESHOLD = 75 
MIN_INSTANCES_PER_GENDER = 20

SKEW_NUM = 3
SKEW_DENOM = 4
UPPERBOUND_CHECK = 200

def collect_dataset(data, man, woman, human_verbs, human_count, output_json):
    # write adjusted dataset
    output = dict()
    print_stats(human_count, human_verbs)
    for image_name in data:
        collect_limited_human_activities(image_name, human_count, human_verbs, man, woman, data, output)

    with open(output_json, "w") as f:     
        json.dump(output, f) 

    # store verbs of humans in action
    with open(args.human_verbs_txt, 'w') as f:
        for verb in human_verbs:
            f.write("%s\n" % verb)

def prepare_balanced_and_skewed_datasets(data, man, woman, human_count, output_json):
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
        # print("Images per Activity", "Activities", "Total Images")
        # print(image_per_verb_used, num_verbs, current)
        # print_border()

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
    output_split = output_json.split("/")
    balanced_output_json = os.path.join(output_split[0], "balanced_"+output_split[1])
    collect_dataset(data, man, woman, human_verbs, balanced_human_count, balanced_output_json)
    skewed_output_json = os.path.join(output_split[0], "skewed_"+output_split[1])
    collect_dataset(data, man, woman, human_verbs, skewed_human_count, skewed_output_json)

def parse_json(args):
    # collect gender encodings
    nouns, verbs = get_space()
    man = get_noun(nouns, MAN)
    woman = get_noun(nouns, WOMAN)

    # collect human activities
    data = json.load(open(args.data_json))
    human_count = count_human_activities(data, man, woman)
    
    # collect datasets to form annotations for 
    if args.balanced_and_skewed:
        datasets = prepare_balanced_and_skewed_datasets(data, man, woman, human_count, args.output_json)
    else:
        human_verbs = [k for k in human_count if sum(human_count[k]) > THRESHOLD and min(human_count[k]) >= MIN_INSTANCES_PER_GENDER]
        datasets = collect_dataset(data, man, woman, human_verbs, human_count, args.output_json)

# Sample execution: 
# python preprocess_train.py data/train.json data/genders_train.json data/human_verbs.txt > stats/gender_train_stats.txt
# python preprocess_train.py data/train.json data/genders_train.json data/balanced_human_verbs.txt --balanced_and_skewed > stats/balanced_gender_train_stats.txt
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Fetch all actions with man or woman agents.") 
  parser.add_argument("data_json", help="Input dataset json to preprocess.") 
  parser.add_argument("output_json", help="Output dataset json to collect.")
  parser.add_argument("human_verbs_txt", help="Txt file to write verbs of humans in action.")
  parser.add_argument("--balanced_and_skewed", action='store_true', default=False, help="set to True to form balanced and skewed datasets")
  args = parser.parse_args()

  parse_json(args)
