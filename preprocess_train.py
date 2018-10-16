import argparse
import json

from imsitu_utils import *

THRESHOLD = 85 
# threshold used in the paper, resulting in 212 verbs
# if we remove images containing both men and women, we get 203 verbs

MIN_INSTANCES_PER_GENDER = 20

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
            # human_count[verb][0] = human_count[verb][0] + 1
            # human_count[verb][1] = human_count[verb][1] + 1
        elif man in agents:
            human_count[verb][0] = human_count[verb][0] + 1
        elif woman in agents:
            human_count[verb][1] = human_count[verb][1] + 1
        
    # output set statistics; check sample stability across different thresholds
    for i in range(THRESHOLD - 20 , THRESHOLD + 20):
        human_verbs = [k for k in human_count if sum(human_count[k]) > THRESHOLD and min(human_count[k]) >= MIN_INSTANCES_PER_GENDER]    
        print_stats(human_count, human_verbs)
    print("------------------------------------------------------------")

    # write verbs of humans in action
    with open(args.human_verbs_txt, 'w') as f:
        for verb in human_verbs:
            f.write("%s\n" % verb)

    # write adjusted dataset
    output = dict()
    human_count = dict()
    for image_name in data:
        collect_human_action(image_name, human_count, human_verbs, man, woman, data, output)

    with open(args.output_json, "w") as f:     
        json.dump(output, f) 

    # sanity check; output set statistics that should match the ones above
    print_stats(human_count, human_verbs)

    # test what we wrote:
    # output = json.load(open(args.output_json))
    # for k in output:
    #     print(output[k])

# Sample execution: python preprocess_train.py data/train.json data/genders_train.json human_verbs.txt
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Fetch all actions with man or woman agents.") 
  parser.add_argument("data_json", help="Input dataset json to preprocess.") 
  parser.add_argument("output_json", help="Output dataset json to collect.")
  parser.add_argument("human_verbs_txt", help="Txt file to write verbs of humans in action.")
  args = parser.parse_args()

  parse_json(args)


# Results for Training Set

# Output with "both genders present" excluded: 
# Total verbs 504
# Total nouns 82115
# Verbs: 175, Images with Man: 12164, Images with Woman: 9374
