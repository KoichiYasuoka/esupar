#! /usr/bin/python -i
# coding=utf-8

MODELS={
  "ja":"KoichiYasuoka/bert-base-japanese-upos",
  "ja_base":"KoichiYasuoka/bert-base-japanese-upos",
  "ja_large":"KoichiYasuoka/bert-large-japanese-upos",
  "ja_luw":"KoichiYasuoka/bert-base-japanese-luw-upos",
  "ja_luw_small":"KoichiYasuoka/roberta-small-japanese-char-luw-upos",
  "ja_luw_base":"KoichiYasuoka/bert-base-japanese-luw-upos",
  "ja_luw_large":"KoichiYasuoka/bert-large-japanese-luw-upos",
  "lzh":"KoichiYasuoka/roberta-classical-chinese-base-upos",
  "lzh_base":"KoichiYasuoka/roberta-classical-chinese-base-upos",
  "lzh_large":"KoichiYasuoka/roberta-classical-chinese-large-upos",
  "th":"KoichiYasuoka/roberta-base-thai-syllable-upos",
  "zh":"KoichiYasuoka/chinese-bert-wwm-ext-upos",
  "zh_bert":"KoichiYasuoka/chinese-bert-wwm-ext-upos",
  "zh_base":"KoichiYasuoka/chinese-roberta-base-upos",
  "zh_large":"KoichiYasuoka/chinese-roberta-large-upos"
}

class Esupar(object):
  def __init__(self,model):
    import os
    from transformers import AutoTokenizer,AutoModelForTokenClassification
    from transformers.file_utils import cached_path,hf_bucket_url
    from supar import Parser
    self.tokenizer=AutoTokenizer.from_pretrained(model)
    self.tokenizerfast=(str(type(self.tokenizer)).find("TokenizerFast")>0)
    self.tagger=AutoModelForTokenClassification.from_pretrained(model)
    f=os.path.join(model,"supar.model")
    if os.path.isfile(f):
      self.parser=Parser.load(f)
    else:
      self.parser=Parser.load(cached_path(hf_bucket_url(model,"supar.model")))
  def __call__(self,sentence):
    import torch
    if self.tokenizerfast:
      v=self.tokenizer(sentence,return_offsets_mapping=True)
    else:
      v=self.mapping(sentence)
    if len(v["input_ids"])<self.tokenizer.model_max_length:
      w=[self.tagger.config.id2label[q] for q in torch.argmax(self.tagger(torch.tensor([v["input_ids"]]))["logits"],dim=2)[0].tolist()]
      x=[[p,s,e] for (s,e),p in zip(v["offset_mapping"],w) if s<e]
    else:
      t=0
      x=[]
      while len(v["input_ids"])>=self.tokenizer.model_max_length:
        w=[self.tagger.config.id2label[q] for q in torch.argmax(self.tagger(torch.tensor([v["input_ids"][0:self.tokenizer.model_max_length-1]]))["logits"],dim=2)[0].tolist()]
        x+=[[p,s+t,e+t] for (s,e),p in zip(v["offset_mapping"][0:self.tokenizer.model_max_length-1],w) if s<e]
        while x[-1][0].startswith("I-"):
          x.pop()
        if x[-1][0].startswith("B-"):
          x.pop()
        t=x[-1][2]
        if self.tokenizerfast:
          v=self.tokenizer(sentence[t:],return_offsets_mapping=True)
        else:
          v=self.mapping(sentence[t:])
      w=[self.tagger.config.id2label[q] for q in torch.argmax(self.tagger(torch.tensor([v["input_ids"]]))["logits"],dim=2)[0].tolist()]
      x+=[[p,s+t,e+t] for (s,e),p in zip(v["offset_mapping"],w) if s<e]
    for i in range(len(x)-1,0,-1):
      if x[i][0].startswith("I-"):
        if x[i-1][0]==x[i][0]:
          p,s,e=x.pop(i)
          x[i-1][2]=e
        elif x[i-1][0]=="B-"+x[i][0][2:]:
          p,s,e=x.pop(i)
          x[i-1]=[x[i-1][0][2:],x[i-1][1],e]
        elif x[i][0].startswith("I-PUNCT"):
          x[i][0]=x[i][0][2:]
        elif x[i-1][0].startswith("B-"):
          p,s,e=x.pop(i)
          x[i-1]=[x[i-1][0][2:],x[i-1][1],e]
        elif x[i-1][0].startswith("I-"):
          p,s,e=x.pop(i)
          x[i-1][2]=e
        elif x[i-1][0]==x[i][0][2:]:
          p,s,e=x.pop(i)
          x[i-1]=[x[i-1][0],x[i-1][1],e]
        elif x[i][0].startswith("I-"+x[i-1][0]+"+"):
          p,s,e=x.pop(i)
          x[i-1]=[p[2:],x[i-1][1],e]
    for i in range(0,len(x)):
      if x[i][0].startswith("B-") or x[i][0].startswith("I-"):
        x[i][0]=x[i][0][2:]
    if ".".join(p for p,s,e in x).find("+")<0:
      d=self.parser.predict([[sentence[s:e] for p,s,e in x]]).sentences[0]
      d.values[3]=tuple([p for p,s,e in x])
      d.values[9]=tuple(["SpaceAfter=No" if e==s else "_" for (_,_,e),(_,s,_) in zip(x,x[1:])]+["_"])
    else:
      try:
        c=self.tagger.config.task_specific_params["upos_multiword"]
      except:
        c={}
      v,m=[],[]
      for i,(p,s,e) in enumerate(x):
        t=sentence[s:e]
        if p.find("+")<0:
          v.append((t,p,"SpaceAfter=No" if i+1<len(x) and e==x[i+1][1] else "_"))
        else:
          q=p.split("+")
          w=[t]+["_"]*len(q)
          if p in c and t in c[p]:
            w=c[p][t]+["_"]*len(q)
          m.append((len(v)+1,len(v)+len(q),t,"SpaceAfter=No" if i+1<len(x) and e==x[i+1][1] else "_"))
          for j,k in zip(w,q):
            v.append((j,k,"_"))
      d=self.parser.predict([[t for t,p,z in v]]).sentences[0]
      d.values[3]=tuple([p for t,p,z in v])
      d.values[9]=tuple([z for t,p,z in v])
      for s,e,t,z in reversed(m):
        for i in range(10):
          x=list(d.values[i])
          if i==0:
            d.values[i]=tuple(x[0:s-1]+[str(s)+"-"+str(e)]+x[s-1:])
          elif i==1:
            d.values[i]=tuple(x[0:s-1]+[t]+x[s-1:])
          elif i==9:
            d.values[i]=tuple(x[0:s-1]+[z]+x[s-1:])
          else:
            d.values[i]=tuple(x[0:s-1]+["_"]+x[s-1:])
    return d
  def mapping(self,sentence):
    import tokenizations
    v=self.tokenizer(sentence)
    a=v["input_ids"]
    x,y=tokenizations.get_alignments(self.tokenizer.convert_ids_to_tokens(a),sentence)
    w=[]
    for i,t in enumerate(x):
      if t>[]:
        s=(t[0],t[-1]+1)
      elif a[i]==self.tokenizer.unk_token_id:
        b=[[-1]]+[t for t in x[0:i] if t>[]]
        e=[t for t in x[i+1:] if t>[]]+[[len(sentence)]]
        s=(b[-1][-1]+1,e[0][0])
      else:
        s=(0,0)
      w.append(s)
    v["offset_mapping"]=w
    return v

def load(model="ja"):
  if model in MODELS:
    model=MODELS[model]
  return Esupar(model)

