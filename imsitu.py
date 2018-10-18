# Code borrowed from: https://github.com/my89/imSitu/blob/master/baseline_crf.py

import torch as torch
import torch.utils.data as data
from PIL import Image
import os
import os.path
import sys
 
class imSituVerbRoleNounEncoder:
  
  def n_verbs(self): return len(self.v_id)
  def n_nouns(self): return len(self.n_id)
  def n_roles(self): return len(self.r_id)
  def verbposition_role(self,v,i): return self.v_r[v][i]
  def verb_nroles(self, v): return len(self.v_r[v])
  def max_roles(self): return self.mr  
  def pad_symbol(self): return -1
  def unk_symbol(self): return -2

  def __init__(self, dataset):
    self.v_id = {}
    self.id_v = {}
   
    self.r_id = {}
    self.id_r = {}

    self.id_n = {}
    self.n_id = {}

    self.mr = 0
 
    self.v_r = {} 

    for (image, annotation) in dataset.items():
      v = annotation["verb"]
      if v not in self.v_id: 
        _id = len(self.v_id)
        self.v_id[v] = _id
        self.id_v[_id] = v
        self.v_r[_id]  = []
      vid = self.v_id[v]
      for frame in annotation["frames"]:
        for (r,n) in frame.items():
          if r not in self.r_id: 
            _id = len(self.r_id)
            self.r_id[r] = _id
            self.id_r[_id] = r

          if n not in self.n_id: 
            _id = len(self.n_id)
            self.n_id[n] = _id
            self.id_n[_id] = n
 
          rid = self.r_id[r]
          if rid not in self.v_r[vid]: self.v_r[vid].append(rid)                    
  
    for (v,rs) in self.v_r.items(): 
      if len(rs) > self.mr : self.mr = len(rs)
    
    for (v, vid) in self.v_id.items():  self.v_r[vid] = sorted(self.v_r[vid])

   
  def encode(self, situation):
    rv = {}
    verb = self.v_id[situation["verb"]]
    rv["verb"] = verb
    rv["frames"] = []
    for frame in situation["frames"]:
      _e = []
      for (r,n) in frame.items():
        if r in self.r_id: _rid = self.r_id[r]
        else: _rid = self.unk_symbol()
        if n in self.n_id: _nid = self.n_id[n]
        else: _nid = self.unk_symbol()
        _e.append((_rid, _nid))
      rv["frames"].append(_e)
    return rv

  def decode(self, situation):
    verb = self.id_v[situation["verb"]]
    rv = {"verb": verb, "frames":[]}
    for frame in situation["frames"]:
      _fr = {}
      for (r,n) in frame.items():
        _fr[self.id_r[r]] =  self.id_n[n]
      rv["frames"].append(_fr)
    return rv     

  # produce a tensor of the batch of situations
  def to_tensor(self, situations):
    rv = []
    for situation in situations:
      _rv = self.encode(situation)

      # append encoded verb label
      verb = _rv["verb"]
      rv.append(torch.LongTensor([verb]))
    return torch.cat(rv)
  
  # the tensor is BATCH x VERB X FRAME
  def to_situation(self, tensor, use_verb_only=True):
    (batch,verbd,_) = tensor.size()
    rv = []
    for b in range(0, batch):
      _tensor = tensor[b]
      for verb in range(0, verbd):
        args = []
        __tensor = _tensor[verb]
        if not use_verb_only:
          for j in range(0, self.verb_nroles(verb)):
            n = __tensor.data[j]
            args.append((self.verbposition_role(verb,j),n))
          situation = {"verb": verb, "frames":[args]}
        else:
          situation = {"verb": verb}
        rv.append(self.decode(situation))
    return rv

class imSituVerbRoleLocalNounEncoder(imSituVerbRoleNounEncoder):
  
  def n_verbrole(self): return len(self.vr_id)
  def n_verbrolenoun(self): return self.total_vrn
  def verbposition_role(self,v,i): return self.v_vr[v][i]
  def verb_nroles(self, v): return len(self.v_vr[v])
 
  def __init__(self, dataset):
    imSituVerbRoleNounEncoder.__init__(self, dataset)
    self.vr_id = {}
    self.id_vr = {}
   
    self.vr_n_id = {}
    self.vr_id_n = {} 

    self.vr_v = {}
    self.v_vr = {}

    self.total_vrn = 0      

    for (image, annotation) in dataset.items():
      v = self.v_id[annotation["verb"]]
  
      for frame in annotation["frames"]:
        for(r,n) in frame.items(): 
          r = self.r_id[r]
          n = self.n_id[n]

          if (v,r) not in self.vr_id:
            _id = len(self.vr_id)
            self.vr_id[(v,r)] = _id
            self.id_vr[_id] = (v,r)
            self.vr_n_id[_id] = {}
            self.vr_id_n[_id] = {}             

          vr = self.vr_id[(v,r)]    
          if v not in self.v_vr: self.v_vr[v] = []
          self.vr_v[vr] = v
          if vr not in self.v_vr[v]: self.v_vr[v].append(vr)
        
          if n not in self.vr_n_id[vr]:
            _id = len(self.vr_n_id[vr]) 
            self.vr_n_id[vr][n] = _id
            self.vr_id_n[vr][_id] = n
            self.total_vrn += 1

  def encode(self, situation):
    v = self.v_id[situation["verb"]]
    rv = {"verb": v, "frames": []}
    for frame in situation["frames"]:
      _e = [] 
      for (r,n) in frame.items():
        if r not in self.r_id: r = self.unk_symbol()
        else: r = self.r_id[r]
        if n not in self.n_id: n = self.unk_symbol()
        else: n = self.n_id[n]
        if (v,r) not in self.vr_id: vr = self.unk_symbol()
        else: vr = self.vr_id[(v,r)]
        if vr not in self.vr_n_id: vrn = self.unk_symbol()
        elif n not in self.vr_n_id[vr]: vrn = self.unk_symbol()
        else: vrn = self.vr_n_id[vr][n]    
        _e.append((vr, vrn))
      rv["frames"].append(_e) 
    return rv

  def decode(self, situation):
    #print situation
    verb = self.id_v[situation["verb"]]
    rv = {"verb": verb, "frames":[]}
    for frame in situation["frames"]:
      _fr = {}
      for (vr,vrn) in frame:
        if vrn not in self.vr_id_n[vr]: 
          print "index error, verb = {}".format(verb)
          n = -1
        else:
          n = self.id_n[self.vr_id_n[vr][vrn]]
        r = self.id_r[self.id_vr[vr][1]]
        _fr[r]=n
      rv["frames"].append(_fr)
    return rv 

class imSituSituation(data.Dataset):
   def __init__(self, root, annotation_file, encoder, transform=None):
        self.root = root
        self.imsitu = annotation_file
        self.ids = list(self.imsitu.keys())
        self.encoder = encoder
        self.transform = transform
   
   def index_image(self, index):
        rv = []
        index = index.view(-1)
        for i in range(index.size()[0]):
          rv.append(self.ids[index[i]])
        return rv
      
   def __getitem__(self, index):
        imsitu = self.imsitu
        _id = self.ids[index]
        ann = self.imsitu[_id]
       
        img = Image.open(os.path.join(self.root, _id)).convert('RGB')
        
        if self.transform is not None: img = self.transform(img)
        target = self.encoder.to_tensor([ann])

        return (torch.LongTensor([index]), img, target)

   def __len__(self):
        return len(self.ids)
