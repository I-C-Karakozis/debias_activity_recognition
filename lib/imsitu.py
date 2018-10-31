import torch as torch
import torch.utils.data as data
from PIL import Image
import os
import os.path
import sys
 
class imSituVerbRoleNounEncoder:
  
  def n_classes(self): return len(self.v_id)
  def n_verbs(self): return len(self.v_id)
  def n_nouns(self): return len(self.n_id)

  def __init__(self, dataset):
    self.v_id = {}
    self.id_v = {}

    self.n_id = {}
    self.id_n = {}
    
    self.vn_id = {}
    self.id_vn = {}

    self.genders = []
    self.verbs = []

    for (image, annotation) in dataset.items():
      # encode verb
      v = annotation["verb"]
      if v not in self.v_id: 
        _id = len(self.v_id)
        self.v_id[v] = _id
        self.id_v[_id] = v
        self.verbs.append(v)

      # encode agent
      n = annotation['frames'][0]['agent']
      if n not in self.n_id: 
        _id = len(self.n_id)
        self.n_id[n] = _id
        self.id_n[_id] = n
        self.genders.append(n)

      # encode verb agent
      vn = v + "_" + n
      if vn not in self.vn_id: 
        _id = len(self.vn_id)
        self.vn_id[v] = _id
        self.id_vn[_id] = vn
   
  def encode(self, situation):
    rv = {}
    rv["verb"]= self.v_id[situation["verb"]]
    rv["agent"] = self.n_id[situation['frames'][0]['agent']]
    return rv

  def encode_verb_noun(self, verb, noun):
    return self.vn_id[verb + "_" + noun]

  def decode_verb(self, v_id):
    return self.id_v[v_id]

  def decode_noun(self, n_id):
    return self.id_n[n_id]

  # produce a tensor of the batch of situations
  def to_tensor(self, situations, gender_cls=False, test=False):
    rv = []
    for situation in situations:
      _rv = self.encode(situation)
      # append encoded target label
      if gender_cls: label = _rv["agent"]
      elif test:     label = (_rv["verb"], _rv["agent"])
      else:          label = _rv["verb"]
      rv.append(torch.LongTensor([label]))
    return torch.cat(rv)

class imSitu2nClassEncoder:

  def __init__(self, dataset):
    self.v_id = {}
    self.id_v = {}

    self.genders = []
    self.verbs = []

    for (image, annotation) in dataset.items():
      verb = annotation["verb"]
      gender = annotation['frames'][0]['agent']
      v = verb + "_" + gender

      if v not in self.v_id: 
        _id = len(self.v_id)
        self.v_id[v] = _id
        self.id_v[_id] = v

      if verb not in self.verbs:
        self.verbs.append(verb)

      if gender not in self.genders:
        self.genders.append(gender)
   
  def n_classes(self): 
    return len(self.v_id)

  def encode(self, situation):
    return self.v_id[situation["verb"] + "_" + situation['frames'][0]['agent']]

  def encode_verb_noun(self, verb, noun):
    return self.encode({'verb': verb, 'frames':[{'agent': noun}]})

  def decode(self, _id):
    return self.id_v[_id]

  def get_verb_id(self, _id):
    v = self.id_v[_id]
    verb = v.split('_')[0]
    return self.verbs.index(verb)

  def get_gender_ids_for_verb(self, _id):
    v = self.id_v[_id]
    verb = v.split('_')[0]
    activity_ids = [self.encode_verb_noun(verb, gender) for gender in self.genders]
    assert(len(activity_ids) == 2)
    return activity_ids

  # produce a tensor of the batch of situations
  def to_tensor(self, situations, gender_cls=False):
    rv = []
    for situation in situations:
      label = self.encode(situation)
      rv.append(torch.LongTensor([label]))
    return torch.cat(rv)

class imSituSituation(data.Dataset):
   def __init__(self, root, annotation_file, encoder, transform=None, gender_cls=False, test=False):
        self.root = root
        self.imsitu = annotation_file
        self.ids = list(self.imsitu.keys())
        self.encoder = encoder
        self.transform = transform
        self.gender_cls = gender_cls
        self.test = test
      
   def __getitem__(self, index):
        imsitu = self.imsitu
        _id = self.ids[index]
        ann = self.imsitu[_id]
       
        img = Image.open(os.path.join(self.root, _id)).convert('RGB')
        
        if self.transform is not None: img = self.transform(img)
        target = self.encoder.to_tensor([ann], gender_cls=self.gender_cls, test=self.test)

        return (torch.LongTensor([index]), img, target)

   def __len__(self):
        return len(self.ids)
