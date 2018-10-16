import argparse
import json

from imsitu_utils import *

def parse_json(args):
    # collect gender encodings
    nouns, verbs = get_space()
    man = get_noun(nouns, MAN)
    woman = get_noun(nouns, WOMAN)

    # get human verbs and data
    human_verbs = open(args.human_verbs_txt).readlines()
    human_verbs = [verb.strip('\n') for verb in human_verbs]
    data = json.load(open(args.data_json))

    # collect correct samples
    print("Original sample count: {0}".format(len(data)))
    output = dict()
    human_count = dict()
    for image_name in data:
        collect_human_action(image_name, human_count, human_verbs, man, woman, data, output)
    print_stats(human_count, human_verbs)
    print("Final sample count: {0}".format(len(output)))
        
    # write adjusted dataset
    with open(args.output_json, "w") as f:     
        json.dump(output, f)        

    # test what we wrote:
    # output = json.load(open(args.output_json))
    # for k in output:
    #     print(output[k])

# Sample execution: 
# python preprocess_test.py human_verbs.txt data/test.json data/genders_test.json 
# python preprocess_test.py human_verbs.txt data/dev.json data/genders_dev.json
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Fetch all actions with man or woman agents.")
  parser.add_argument("human_verbs_txt", help="Txt file to get verbs of humans in action.")
  parser.add_argument("data_json", help="Input dataset json to preprocess.") 
  parser.add_argument("output_json", help="Output dataset json to collect.")
  args = parser.parse_args()

  parse_json(args)


# Results for Test Set

# Output with "both genders present" excluded: 
# Original sample count: 25200
# Verbs: 175, Images with Man: 3377, Images with Woman: 2509
# Final sample count: 5886

# Results for Dev Set

# Output with "both genders present" excluded: 
# Original sample count: 25200
# Verbs: 175, Images with Man: 3283, Images with Woman: 2651
# Final sample count: 5934
