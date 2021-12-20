#! /usr/bin/python -i
# coding=utf-8

MODELS={
  "ja":"KoichiYasuoka/bert-base-japanese-upos",
  "ja_base":"KoichiYasuoka/bert-base-japanese-upos",
  "ja_large":"KoichiYasuoka/bert-large-japanese-upos",
  "ja_luw":"KoichiYasuoka/bert-base-japanese-luw-upos",
  "ja_luw_small":"KoichiYasuoka/roberta-small-japanese-luw-upos",
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
    from transformers import AutoTokenizer,AutoModelForTokenClassification
    from transformers.file_utils import cached_path,hf_bucket_url
    from supar import Parser
    self.tokenizer=AutoTokenizer.from_pretrained(model)
    self.tagger=AutoModelForTokenClassification.from_pretrained(model)
    self.parser=Parser.load(cached_path(hf_bucket_url(model,"supar.model")))
  def __call__(self,sentence):
    import torch
    v=self.tokenizer(sentence,return_offsets_mapping=True)
    w=[self.tagger.config.id2label[q] for q in torch.argmax(self.tagger(torch.tensor([v["input_ids"]]))[0],dim=2)[0].tolist()]
    x=[[p,s,e] for (s,e),p in zip(v["offset_mapping"],w) if s<e]
    for i in range(len(x)-1,0,-1):
      if x[i][0].startswith("I-"):
        if x[i-1][0].startswith("B-"):
          p,s,e=x.pop(i)
          x[i-1]=[x[i-1][0][2:],x[i-1][1],e]
        elif x[i-1][0].startswith("I-"):
          p,s,e=x.pop(i)
          x[i-1][2]=e
    for i in range(0,len(x)):
      if x[i][0].startswith("B-") or x[i][0].startswith("I-"):
        x[i][0]=x[i][0][2:]
    d=self.parser.predict([[sentence[s:e] for p,s,e in x]]).sentences[0]
    d.values[3]=tuple([p for p,s,e in x])
    d.values[9]=tuple(["SpaceAfter=No" if e==s else "_" for (_,_,e),(_,s,_) in zip(x,x[1:])]+["_"])
    return d

def load(model="ja"):
  if model in MODELS:
    model=MODELS[model]
  return Esupar(model)

