import argparse
import json

from lib.imsitu_utils import *

def parse_json(args):
    # collect gender encodings
    nouns, verbs = get_space()
    man = get_noun(nouns, MAN)
    woman = get_noun(nouns, WOMAN)

    # get human verbs and data
    human_verbs = open(args.human_verbs_txt).readlines()
    human_verbs = [verb.strip('\n') for verb in human_verbs]
    data = json.load(open(args.data_json))
    print("Original sample count: {0}".format(len(data)))

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
        print_stats(human_count, updated_human_verbs)
        for image_name in data:
            collect_limited_human_activities(image_name, human_count, human_verbs, man, woman, data, output)        
    else:
        human_count = dict()
        for image_name in data:
            collect_all_human_activities(image_name, human_count, human_verbs, man, woman, data, output)
        print_stats(human_count, human_verbs)
        
    # write adjusted dataset
    with open(args.output_json, "w") as f:     
        json.dump(output, f)        

    # test what we wrote:
    # output = json.load(open(args.output_json))
    # for k in output:
    #     print(output[k])

# Sample execution: 

# python preprocess_test.py data/human_verbs.txt data/test.json data/genders_test.json > stats/gender_test_stats.txt
# python preprocess_test.py data/activity_balanced_human_verbs.txt data/test.json data/activity_balanced_test.json > stats/activity_balanced_test_stats.txt
# python preprocess_test.py data/balanced_human_verbs.txt data/test.json data/skewed_genders_test.json > stats/skewed_gender_test_stats.txt
# python preprocess_test.py data/balanced_human_verbs.txt data/test.json data/balanced_genders_test.json --balanced > stats/balanced_gender_test_stats.txt

# python preprocess_test.py data/human_verbs.txt data/dev.json data/genders_dev.json > stats/gender_dev_stats.txt
# python preprocess_test.py data/activity_balanced_human_verbs.txt data/dev.json data/activity_balanced_dev.json > stats/activity_balanced_dev_stats.txt
# python preprocess_test.py data/balanced_human_verbs.txt data/test.json data/skewed_genders_dev.json > stats/skewed_gender_dev_stats.txt
# python preprocess_test.py data/balanced_human_verbs.txt data/dev.json data/balanced_genders_dev.json --balanced > stats/balanced_gender_dev_stats.txt

# python preprocess_test.py data/human_verbs.txt data/concat_test.json data/genders_concat_test.json > stats/gender_concat_test_stats.txt
# python preprocess_test.py data/balanced_human_verbs.txt data/concat_test.json data/balanced_genders_concat_test.json --balanced > stats/balanced_gender_concat_test_stats.txt

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Fetch all actions with man or woman agents.")
  parser.add_argument("human_verbs_txt", help="Txt file to get verbs of humans in action.")
  parser.add_argument("data_json", help="Input dataset json to preprocess.") 
  parser.add_argument("output_json", help="Output dataset json to collect.")
  parser.add_argument("--balanced", action='store_true', default=False, help="set to True to form gender balanced dataset")
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
