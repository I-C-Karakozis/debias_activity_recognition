import json

MAN = "adult male"
WOMAN = "adult female"

################### Functions for Getting imSitu Elements ###################
def get_space():
    # load imsitu space
    imsitu = json.load(open("data/imsitu_space.json"))
    nouns = imsitu["nouns"]
    verbs = imsitu["verbs"]

    # print("Total verbs {0}".format(len(verbs)))
    # print("Total nouns {0}".format(len(nouns)))

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
    ''' Maintain only verb and agent information '''
    image['frames'] = [frame for frame in image['frames'] if agent == frame['agent']]
    frame = image['frames'][0]
    image['frames'] = [{'agent': frame['agent']}]
    return image

################### Functions for Logging ###################

def print_border():
    print("-" * 20)

def print_noun_entries(nouns, target):
    for noun in nouns:
        if target in nouns[noun]["gloss"]:
            print(target, noun, nouns[noun])

def print_stats(human_count, human_verbs):
    print_border()

    # Print Distribution per Verb
    print("Per Verb Satistics:")
    print("{: >20} {: >20} {: >20} {: >20}".format(*["Activity","Total Images","Man Conc.","Woman Conc."]))
    for verb in human_verbs:
        total  = sum(human_count[verb])
        m_perc = round(human_count[verb][0] / float(total) * 100, 2)
        w_perc = 100 - m_perc
        print("{: >20} {: >20} {: >20}% {: >20}%".format(*[verb, total, m_perc, w_perc]))
    print_border()

    # Print Aggregate Stats
    print("Aggregate Satistics:")
    man_count = sum(human_count[k][0] for k in human_verbs)
    woman_count = sum(human_count[k][1] for k in human_verbs)
    print("Verbs: {0}, Images with Man: {1}, Images with Woman: {2}".format(len(human_verbs), man_count, woman_count))
    print("Total Image Count: {0}".format(man_count + woman_count))
    print_border()

################### Functions for Preprocessing Human Activities ###################

def count_human_activities(data, man, woman):
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

    return human_count

def collect_limited_human_activities(image_name, human_count, human_verbs, man, woman, data, output):
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

def collect_all_human_activities(image_name, human_count, human_verbs, man, woman, data, output):
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
