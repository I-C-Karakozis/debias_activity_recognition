import argparse
import json

from imsitu_utils import *

# TODO: concatenate train and dev for evaluation

# threshold used in the paper: 85 (resulting in 212 verbs)
# if we remove images containing both men and women, we get 203 verbs
THRESHOLD = 85 
MIN_INSTANCES_PER_GENDER = 20

SKEW = 0.75
SKEW_DENOM = 4

def collect_limited_human_action(image_name, human_count, human_verbs, man, woman, data, output):
    image = data[image_name]

    # validate human verb
    verb = image["verb"]
    if verb not in human_verbs: 
        return
    
    # collect all agents of action in the image
    agents = get_agents(image)

    # count humans in action and record gender distribution
    if man in agents and woman in agents:
        return
    elif human_count[verb][0] > 0 and man in agents:
        human_count[verb][0] = human_count[verb][0] - 1
        output[image_name] = get_frame(image, man)
    elif human_count[verb][1] > 0 and woman in agents:
        human_count[verb][1] = human_count[verb][1] - 1
        output[image_name] = get_frame(image, woman)

def parse_json(args):
    # collect gender encodings
    nouns, verbs = get_space()
    man = get_noun(nouns, MAN)
    woman = get_noun(nouns, WOMAN)

    # count number of human agents per verb
    data = json.load(open(args.data_json))
    human_count = dict()
    for image_name in data:
        # extract image metadata
        image = data[image_name]
        verb = image["verb"]
        if verb not in human_count: 
            human_count[verb] = [0 for i in range(2)];
        agents = get_agents(image)

        # count humans in action and record gender distribution
        if man in agents and woman in agents:
            continue
        elif man in agents:
            human_count[verb][0] = human_count[verb][0] + 1
        elif woman in agents:
            human_count[verb][1] = human_count[verb][1] + 1
        
    # output set statistics
    human_verbs = [k for k in human_count if sum(human_count[k]) > THRESHOLD and min(human_count[k]) >= MIN_INSTANCES_PER_GENDER] 
    if args.balanced:
        for verb in human_verbs:
            human_count[verb][0] = min(human_count[verb][0], human_count[verb][1])
            human_count[verb][1] = human_count[verb][0]
    # if args.skewed:  
    #     print("skewed")
    #     for verb in human_verbs:
    #         # identify direction to skew towards
    #         if human_count[verb][0] > human_count[verb][1]: hi = 0; lo = 1
    #         else: hi = 1; lo = 0

    #         # setup to enforce 
    #         total = float(sum(human_count[verb]))
    #         while total % SKEW_DENOM != 0:
    #             if human_count[verb][hi] / total > SKEW:
    #                 human_count[verb][hi] = human_count[verb][hi] - 1
    #             else:
    #                 human_count[verb][lo] = human_count[verb][lo] - 1
    #             total = float(sum(human_count[verb]))
    print_stats(human_count, human_verbs)

    # write verbs of humans in action
    with open(args.human_verbs_txt, 'w') as f:
        for verb in human_verbs:
            f.write("%s\n" % verb)

    # write adjusted dataset
    output = dict()
    for image_name in data:
        collect_limited_human_action(image_name, human_count, human_verbs, man, woman, data, output)

    with open(args.output_json, "w") as f:     
        json.dump(output, f) 

    # test what we wrote:
    # output = json.load(open(args.output_json))
    # for k in output:
    #     print(output[k])

# Sample execution: 
# python preprocess_train.py data/train.json data/genders_train.json data/human_verbs.txt > stats/gender_train_stats.txt
# python preprocess_train.py data/train.json data/balanced_genders_train.json data/balanced_human_verbs.txt --balanced > stats/balanced_gender_train_stats.txt
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Fetch all actions with man or woman agents.") 
  parser.add_argument("data_json", help="Input dataset json to preprocess.") 
  parser.add_argument("output_json", help="Output dataset json to collect.")
  parser.add_argument("human_verbs_txt", help="Txt file to write verbs of humans in action.")
  parser.add_argument("--balanced", action='store_true', default=False, help="set to True to form balanced dataset")
  parser.add_argument("--skewed", action='store_true', default=False, help="set to True to form skewed dataset")
  args = parser.parse_args()

  parse_json(args)


# Results for Training Set

# Output with "both genders present" excluded: 
# Total verbs 504
# Total nouns 82115
# Verbs: 175, Images with Man: 12164, Images with Woman: 9374
# Total Image Count: 21538

# Output for balanced dataset: 
# Total verbs 504
# Total nouns 82115
# Verbs: 175, Images with Man: 7392, Images with Woman: 7392
# Total Image Count: 14784
