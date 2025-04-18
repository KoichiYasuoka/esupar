#! /usr/bin/python -i
# coding=utf-8

MODELS={
  "ain":"KoichiYasuoka/roberta-base-ainu-upos",
  "cop":"KoichiYasuoka/roberta-base-coptic-upos",
  "de":"KoichiYasuoka/bert-base-german-upos",
  "de_base":"KoichiYasuoka/bert-base-german-upos",
  "de_large":"KoichiYasuoka/bert-large-german-upos",
  "en":"KoichiYasuoka/roberta-base-english-upos",
  "en_base":"KoichiYasuoka/roberta-base-english-upos",
  "en_large":"KoichiYasuoka/roberta-large-english-upos",
  "ja":"KoichiYasuoka/bert-base-japanese-upos",
  "ja_base":"KoichiYasuoka/bert-base-japanese-upos",
  "ja_large":"KoichiYasuoka/bert-large-japanese-upos",
  "ja_luw":"KoichiYasuoka/bert-base-japanese-luw-upos",
  "ja_luw_small":"KoichiYasuoka/roberta-small-japanese-char-luw-upos",
  "ja_luw_base":"KoichiYasuoka/bert-base-japanese-luw-upos",
  "ja_luw_large":"KoichiYasuoka/bert-large-japanese-luw-upos",
  "ko":"KoichiYasuoka/roberta-base-korean-upos",
  "ko_base":"KoichiYasuoka/roberta-base-korean-upos",
  "ko_large":"KoichiYasuoka/roberta-large-korean-upos",
  "ko_morph":"KoichiYasuoka/roberta-base-korean-morph-upos",
  "ko_morph_base":"KoichiYasuoka/roberta-base-korean-morph-upos",
  "ko_morph_large":"KoichiYasuoka/roberta-large-korean-morph-upos",
  "lzh":"KoichiYasuoka/roberta-classical-chinese-base-upos",
  "lzh_base":"KoichiYasuoka/roberta-classical-chinese-base-upos",
  "lzh_large":"KoichiYasuoka/roberta-classical-chinese-large-upos",
  "sr":"KoichiYasuoka/gpt2-small-serbian-upos",
  "sr_large":"KoichiYasuoka/gpt2-large-serbian-upos",
  "th":"KoichiYasuoka/roberta-base-thai-spm-upos",
  "vi":"KoichiYasuoka/bert-base-vietnamese-upos",
  "zh":"KoichiYasuoka/chinese-bert-wwm-ext-upos",
  "zh_bert":"KoichiYasuoka/chinese-bert-wwm-ext-upos",
  "zh_base":"KoichiYasuoka/chinese-roberta-base-upos",
  "zh_large":"KoichiYasuoka/chinese-roberta-large-upos"
}

class Esupar(object):
  def __init__(self,model,lemma=None):
    import os,numpy
    from transformers import AutoTokenizer,AutoModelForTokenClassification
    from esupar.supar import Parser
    try:
      from transformers.utils import cached_file
    except:
      from transformers.file_utils import cached_path,hf_bucket_url
      cached_file=lambda x,y:cached_path(hf_bucket_url(x,y))
    self.tokenizer=AutoTokenizer.from_pretrained(model)
    self.tokenizerfast=(str(type(self.tokenizer)).find("TokenizerFast")>0)
    if not self.tokenizerfast:
      try:
        if self.tokenizer.word_tokenizer_type=="mecab":
          from esupar.mecab import BertMecabTokenizerFast
          self.tokenizer=BertMecabTokenizerFast.from_pretrained(model)
          self.tokenizerfast=True
      except:
        pass
    self.tagger=AutoModelForTokenClassification.from_pretrained(model)
    f=os.path.join(model,"supar.model")
    if os.path.isfile(f):
      self.parser=Parser.load(f)
    else:
      self.parser=Parser.load(cached_file(model,"supar.model"))
    try:
      for x in self.parser.transform.flattened_fields:
        if x.fn:
          x.fn=lambda t:" "+t
    except:
      pass
    x=self.tagger.config.id2label
    self.labelmatrix=numpy.full((len(x),len(x)),numpy.nan)
    d=numpy.array([numpy.nan if x[i].startswith("I-") else 0 for i in range(len(x))])
    for i in range(len(x)):
      if x[i].startswith("B-"):
        try:
          self.labelmatrix[i,self.tagger.config.label2id["I-"+x[i][2:]]]=0
        except:
          self.labelmatrix[i]=0
      else:
        self.labelmatrix[i]=d
        if x[i].startswith("I-"):
          self.labelmatrix[i,i]=0
    if not lemma in {"copy","tradify","simplify","hangul","ainu","none"}:
      try:
        lemma=self.tagger.config.task_specific_params["esupar_lemmatize"]
      except:
        pass
    if lemma=="copy":
      self.lemma=lambda x:x
    elif lemma=="tradify":
      from esupar.tradify import tradify
      self.lemma=lambda x:"".join(tradify[c] if c in tradify else c for c in x)
    elif lemma=="simplify":
      from esupar.simplify import simplify
      self.lemma=lambda x:"".join(simplify[c] if c in simplify else c for c in x)
    elif lemma=="hangul":
      from esupar.hangul import hangul
      self.hangul={}
      for k,v in hangul.items():
        for c in "".join(v).replace("(","").replace(")",""):
          self.hangul[c]=k
      self.lemma=lambda x:"".join(self.hangul[c] if c in self.hangul else c for c in x)
    elif lemma=="ainu":
      from esupar.ainu import Lemmatize
      self.lemma=Lemmatize()
    else:
      self.lemma=lambda x:"_"
  def __call__(self,sentence):
    import torch,unicodedata
    if self.tokenizerfast:
      v=self.tokenizer(sentence,return_offsets_mapping=True)
    else:
      v=self.mapping(sentence)
    with torch.no_grad():
      if len(v["input_ids"])==0:
        x=[]
      elif len(v["input_ids"])<self.tokenizer.model_max_length:
        import numpy
        m=self.tagger(torch.tensor([v["input_ids"]])).logits[0].numpy()
        for i in range(len(v["offset_mapping"])-1,0,-1):
          m[i-1]+=numpy.nanmax(m[i]+self.labelmatrix,axis=1)
        w=[numpy.nanargmax(m[0])]
        for i in range(1,len(v["offset_mapping"])):
          w.append(numpy.nanargmax(m[i]+self.labelmatrix[w[-1]]))
        x=[[self.tagger.config.id2label[p],s,e] for (s,e),p in zip(v["offset_mapping"],w) if s<e]
      else:
        t=0
        x=[]
        while len(v["input_ids"])>=self.tokenizer.model_max_length:
          w=[self.tagger.config.id2label[q] for q in torch.argmax(self.tagger(torch.tensor([v["input_ids"][0:self.tokenizer.model_max_length-1]])).logits,dim=2)[0].tolist()]
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
        w=[self.tagger.config.id2label[q] for q in torch.argmax(self.tagger(torch.tensor([v["input_ids"]])).logits,dim=2)[0].tolist()]
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
    if ".".join(p for p,_,_ in x).find("+")<0 and [1 for (_,_,e),(_,s,_) in zip(x,x[1:]) if e>s]==[]:
      v=[]
      for p,s,e in x:
        if unicodedata.category(sentence[s])=="Mn" and v!=[]:
          v[-1]=v[-1][0:-1]
          v.append(sentence[s-1:e].replace("\n"," "))
        else:
          v.append(sentence[s:e].replace("\n"," "))
      d=self.parser.predict([v]).sentences[0]
      v=[p.split("|") for p,s,e in x]
      d.values[2]=tuple([self.lemma(f) for f in d.values[1]])
      d.values[3]=tuple(["AUX" if q in {"aux","cop"} and p[0] in {"VERB","ADJ","ADV"} else p[0] for p,q in zip(v,d.values[7])])
      d.values[4]=tuple(["_" if len(p)<2 or p[1].find("=")>0 else p[1] for p in v])
      d.values[5]=tuple(["|".join([f for f in p if f.find("=")>0]+["_"]).replace("|_","") for p in v])
      d.values[9]=tuple(["SpaceAfter=No" if e==s else "_" for (_,_,e),(_,s,_) in zip(x,x[1:])]+["SpaceAfter=No"])
    else:
      try:
        c=self.tagger.config.task_specific_params["upos_multiword"]
      except:
        c={}
      v,m,d=[],[],1
      for i,(p,s,e) in enumerate(x):
        t=sentence[s-1:e] if unicodedata.category(sentence[s])=="Mn" else sentence[s:e]
        if p.find("+")<0:
          if i+1<len(x) and e>x[i+1][1]:
            try:
              j,k=self.lemma.divide(sentence[x[i+1][1]:e])
              t=sentence[s:x[i+1][1]]+j
            except:
              pass
            if i==0 or s>=x[i-1][2]:
              j=i+1
              while j+1<len(x) and x[j][2]>x[j+1][1]:
                j=j+1
              m.append((i+d,j+d,sentence[s:x[j][2]],"_" if j+1<len(x) and x[j][2]<x[j+1][1] else "SpaceAfter=No"))
          if i>0 and s<x[i-1][2]:
            try:
              j,k=self.lemma.divide(sentence[s:x[i-1][2]])
              t=k+t[x[i-1][2]-s:]
            except:
              pass
          v.append((t,p,"_" if i+1<len(x) and e<x[i+1][1] else "SpaceAfter=No"))
        else:
          q=p.split("+")
          d+=len(q)-1
          if p in c and t in c[p]:
            w=c[p][t]+["_"]*len(q)
          else:
            w=[t]+["_"]*len(q)
            if len(t)>len(q):
              w=[t[0:1-len(q)]]+[c for c in t[1-len(q):]]+["_"]*len(q)
          if t=="".join(w[0:len(q)]):
            for j,k in zip(w,q):
              v.append((j,k,"SpaceAfter=No"))
            v[-1]=(j,k,"_" if i+1<len(x) and e<x[i+1][1] else "SpaceAfter=No")
          else:
            m.append((len(v)+1,len(v)+len(q),t,"_" if i+1<len(x) and e<x[i+1][1] else "SpaceAfter=No"))
            for j,k in zip(w,q):
              v.append((j,k,"_"))
      d=self.parser.predict([[t.replace("\n"," ") for t,p,z in v]]).sentences[0]
      d.values[2]=tuple([self.lemma(f) for f in d.values[1]])
      x=[p.split("|") for t,p,z in v]
      d.values[3]=tuple(["AUX" if q in {"aux","cop"} and p[0] in {"VERB","ADJ","ADV"} else p[0] for p,q in zip(x,d.values[7])])
      d.values[4]=tuple(["_" if len(p)<2 or p[1].find("=")>0 else p[1] for p in x])
      d.values[5]=tuple(["|".join([f for f in p if f.find("=")>0]+["_"]).replace("|_","") for p in x])
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
    try:
      import tokenizations
    except:
      import spacy_alignments as tokenizations
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

def load(model="ja",lemma=None):
  if model in MODELS:
    model=MODELS[model]
  return Esupar(model,lemma)

