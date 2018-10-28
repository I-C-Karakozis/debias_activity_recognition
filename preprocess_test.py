import argparse
import json
import os
import sys

from lib.imsitu_utils import *

def parse_json(args):
    # collect gender encodings
    nouns, verbs = get_space()
    man = get_noun(nouns, MAN)
    woman = get_noun(nouns, WOMAN)

    # get human verbs and data
    human_verbs_txt = os.path.join("verbs", args.output_prefix+".txt")
    human_verbs = open(human_verbs_txt).readlines()
    human_verbs = [verb.strip('\n') for verb in human_verbs]
    data = json.load(open(args.data_json))

    # collect correct number of samples amnd activities
    output = dict()
    if args.balanced:
        # enforce gender balance
        human_count = count_human_activities(data, man, woman)
        updated_human_verbs = []
        for verb in human_verbs:
            human_count[verb][0] = min(human_count[verb][0], human_count[verb][1])
            human_count[verb][1] = human_count[verb][0]
            if human_count[verb][0] > 0: updated_human_verbs.append(verb)

        # collect dataset annotations
        sys.stdout = open(os.path.join("stats", args.output_prefix+"_"+args.data_json.split("/")[-1].split(".")[0]+".txt"), "w")
        print_stats(human_count, updated_human_verbs)
        for image_name in data:
            collect_limited_human_activities(image_name, human_count, human_verbs, man, woman, data, output)

    elif args.men_only:
        human_count = count_human_activities(data, man, woman)
        updated_human_verbs = []
        for verb in human_verbs:
            human_count[verb][1] = 0
            if human_count[verb][0] > 0: updated_human_verbs.append(verb)

        # collect dataset annotations
        args.output_prefix += "_men"
        sys.stdout = open(os.path.join("stats", args.output_prefix+"_"+args.data_json.split("/")[-1].split(".")[0]+".txt"), "w")
        print_stats(human_count, updated_human_verbs)
        for image_name in data:
            collect_limited_human_activities(image_name, human_count, human_verbs, man, woman, data, output)         

    elif args.women_only:
        human_count = count_human_activities(data, man, woman)
        updated_human_verbs = []
        for verb in human_verbs:
            human_count[verb][0] = 0
            if human_count[verb][1] > 0: updated_human_verbs.append(verb)

        # collect dataset annotations
        args.output_prefix += "_women"
        sys.stdout = open(os.path.join("stats", args.output_prefix+"_"+args.data_json.split("/")[-1].split(".")[0]+".txt"), "w")
        print_stats(human_count, updated_human_verbs)
        for image_name in data:
            collect_limited_human_activities(image_name, human_count, human_verbs, man, woman, data, output)

    else:
        human_count = dict()
        for image_name in data:
            collect_all_human_activities(image_name, human_count, human_verbs, man, woman, data, output)
        sys.stdout = open(os.path.join("stats", args.output_prefix+"_"+args.data_json.split("/")[-1].split(".")[0]+".txt"), "w")
        print_stats(human_count, human_verbs)
        
    # write adjusted dataset
    dataset = args.data_json.split("/")[-1]
    with open(os.path.join("data", args.output_prefix+"_"+dataset), "w") as f:     
        json.dump(output, f)

# Sample execution: 

# python preprocess_test.py imsitu_data/test.json gender 
# python preprocess_test.py imsitu_data/test.json activity_balanced 
# python preprocess_test.py imsitu_data/test.json activity_balanced --men_only 
# python preprocess_test.py imsitu_data/test.json activity_balanced --women_only 
# python preprocess_test.py imsitu_data/test.json balanced_fixed_gender_ratio 
# python preprocess_test.py imsitu_data/test.json skewed_fixed_gender_ratio 

# python preprocess_test.py imsitu_data/dev.json gender 
# python preprocess_test.py imsitu_data/dev.json activity_balanced 
# python preprocess_test.py imsitu_data/dev.json balanced_fixed_gender_ratio 
# python preprocess_test.py imsitu_data/dev.json skewed_fixed_gender_ratio 

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Fetch all actions with man or woman agents.")
  parser.add_argument("data_json", help="Input dataset json to preprocess.") 
  parser.add_argument("output_prefix", help="Output dataset json to collect.")
  parser.add_argument("--balanced", action='store_true', default=False, help="set to True to form gender balanced dataset")
  parser.add_argument("--men_only", action='store_true', default=False, help="set to True to collect only samples of men")
  parser.add_argument("--women_only", action='store_true', default=False, help="set to True to collect only samples of women")
  args = parser.parse_args()

  parse_json(args)

# Orig Dev Set
# Verbs: 175, Images with Man: 3283, Images with Woman: 2651
# Final sample count: 5934

# Orig Test Set
# Verbs: 175, Images with Man: 3377, Images with Woman: 2509
# Final sample count: 5886

# Balanced Dev Set
# Verbs: 86, Images with Man: 1106, Images with Woman: 1106
# Total Image Count: 2212

# Balanced Test Set
# Verbs: 86, Images with Man: 1110, Images with Woman: 1110
# Total Image Count: 2220

# Skewed Test Set
# Verbs: 86, Images with Man: 1626, Images with Woman: 1379
# Total Image Count: 3005
