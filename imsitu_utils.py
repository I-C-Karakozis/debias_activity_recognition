import json

MAN = "adult male"
WOMAN = "adult female"

def get_space():
    # load imsitu space
    imsitu = json.load(open("data/imsitu_space.json"))
    nouns = imsitu["nouns"]
    verbs = imsitu["verbs"]

    print("Total verbs {0}".format(len(verbs)))
    print("Total nouns {0}".format(len(nouns)))

    return nouns, verbs

def get_noun(nouns, target):
    for noun in nouns:
        if target in nouns[noun]["gloss"]:
            return noun

def get_agents(image):
    agents = [] 
    for frame in image["frames"]:
        if "agent" in frame and len(frame["agent"]) > 0:
            agents.append(frame["agent"])
    return agents

def get_frame(image, agent):
    image['frames'] = [frame for frame in image['frames'] if agent == frame['agent']]
    image['frames'] = [image['frames'][0]]
    return image

def print_noun_entries(nouns, target):
    for noun in nouns:
        if target in nouns[noun]["gloss"]:
            print(target, noun, nouns[noun])

def print_stats(human_count, human_verbs):
    man_count = sum(human_count[k][0] for k in human_verbs)
    woman_count = sum(human_count[k][1] for k in human_verbs)
    print("Verbs: {0}, Images with Man: {1}, Images with Woman: {2}".format(len(human_verbs), man_count, woman_count))
     print("-" * 20)


def collect_human_action(image_name, human_count, human_verbs, man, woman, data, output):
    image = data[image_name]

    # validate human verb
    verb = image["verb"]
    if verb not in human_verbs: 
        return
    if verb not in human_count: 
        human_count[verb] = [0 for i in range(2)];
    
    # collect all agents of action in the image
    agents = get_agents(image)

    # count humans in action and record gender distribution
    if man in agents and woman in agents:
        return
    elif man in agents:
        human_count[verb][0] = human_count[verb][0] + 1
        output[image_name] = get_frame(image, man)
    elif woman in agents:
        human_count[verb][1] = human_count[verb][1] + 1
        output[image_name] = get_frame(image, woman)
