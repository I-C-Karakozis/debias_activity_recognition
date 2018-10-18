import torch as torch
import torch.utils.data as data
from PIL import Image
import os
import os.path
import sys
 
class imSituVerbRoleNounEncoder:
  
  def n_verbs(self): return len(self.v_id)
  def n_nouns(self): return len(self.n_id)

  def __init__(self, dataset):
    self.v_id = {}
    self.id_v = {}

    self.id_n = {}
    self.n_id = {}

    for (image, annotation) in dataset.items():
      # encode verb
      v = annotation["verb"]
      if v not in self.v_id: 
        _id = len(self.v_id)
        self.v_id[v] = _id
        self.id_v[_id] = v

      # encode agents
      n = annotation['frames'][0]['agent']
      if n not in self.n_id: 
        _id = len(self.n_id)
        self.n_id[n] = _id
        self.id_n[_id] = n
   
  def encode(self, situation):
    rv = {}
    rv["verb"]= self.v_id[situation["verb"]]
    rv["agent"] = self.n_id[situation['frames'][0]['agent']]
    return rv   

  # produce a tensor of the batch of situations
  def to_tensor(self, situations, gender_cls=False):
    rv = []
    for situation in situations:
      _rv = self.encode(situation)
      # append encoded target label
      if gender_cls: label = _rv["agent"]
      else:          label = _rv["verb"]
      rv.append(torch.LongTensor([label]))
    return torch.cat(rv)

class imSituSituation(data.Dataset):
   def __init__(self, root, annotation_file, encoder, transform=None, gender_cls=False):
        self.root = root
        self.imsitu = annotation_file
        self.ids = list(self.imsitu.keys())
        self.encoder = encoder
        self.transform = transform
        self.gender_cls = gender_cls
      
   def __getitem__(self, index):
        imsitu = self.imsitu
        _id = self.ids[index]
        ann = self.imsitu[_id]
       
        img = Image.open(os.path.join(self.root, _id)).convert('RGB')
        
        if self.transform is not None: img = self.transform(img)
        target = self.encoder.to_tensor([ann], gender_cls=self.gender_cls)

        return (torch.LongTensor([index]), img, target)

   def __len__(self):
        return len(self.ids)
