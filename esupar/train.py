#! /usr/bin/python -i
# coding=utf-8

import os,sys,subprocess,tempfile

class UPOSDataset(object):
  def __init__(self,conllu,tokenizer):
    self.tokenizerfast=(str(type(tokenizer)).find("TokenizerFast")>0)
    self.ids=[]
    self.upos=[]
    self.multiword={}
    label={"SYM"}
    try:
      from tokenizations import get_alignments
    except:
      get_alignments=None
    with open(conllu,"r",encoding="utf-8") as f:
      form,upos,nosp,mw,mwid,text=[],[],[False],[],[],None
      for t in f:
        w=t.split("\t")
        if len(w)==10:
          if w[0].isdecimal():
            form.append(w[1])
            upos.append(self.getpos(w))
            nosp.append(w[9].find("SpaceAfter=No")>=0)
          elif w[0].find("-")>0:
            mw.append(w[1])
            mwid.append(w[0].split("-")+[w[9].find("SpaceAfter=No")>=0])
        elif t.startswith("# text = "):
          text=t[9:].strip()
        elif t.strip()=="" and form!=[]:
          g={}
          if mw==[] and get_alignments and text:
            v=tokenizer.tokenize(text,add_special_tokens=False)
            w=tokenizer.convert_tokens_to_ids(v)
            k,y=-1,""
            for i,j in enumerate(get_alignments(form,v)[0]):
              if j==[]:
                g={}
                break
              g[i]=[w[x] for x in j]
              if k==j[0]:
                if y=="":
                  y=form[i-1]+form[i]
                  n=[i,i+1]
                else:
                  y+=form[i]
                  n.append(i+1)
              elif y!="":
                mw.append(y)
                mwid.append([n[0],n[-1],nosp[n[-1]]])
                y=""
              k=j[-1]
          if mw!=[]:
            g={}
            if self.tokenizerfast:
              v=tokenizer(mw,add_special_tokens=False,return_offsets_mapping=True)
              m=v["offset_mapping"]
              n=v["input_ids"]
            elif get_alignments:
              m,n,v=[],[],[]
              for i,j in enumerate(tokenizer(mw,add_special_tokens=False)["input_ids"]):
                if [k for k in j if k==tokenizer.unk_token_id]==[]:
                  m.append([(t[0],t[-1]+1) for t in get_alignments(tokenizer.tokenize(mw[i],add_special_tokens=False),mw[i])[0]])
                  n.append(j)
                else:
                  v.append(i)
              for i in reversed(v):
                mw.pop(i)
                mwid.pop(i)
            else:
              m=n=[]
            for i,j,k,l in reversed(list(zip(mw,mwid,m,n))):
              s,e,x,y=int(j[0])-1,int(j[1]),[e for s,e in k],""
              for u in form[s:e]:
                y+=u
                if len(y) not in x:
                  y=""
                  break
              if y!=i:
                p="+".join(upos[s:e])
                if p in self.multiword:
                  self.multiword[p][i]=form[s:e]
                else:
                  self.multiword[p]={i:form[s:e]}
                form=form[0:s]+[i]+form[e:]
                upos=upos[0:s]+[p]+upos[e:]
                nosp=nosp[0:s+1]+[j[2]]+nosp[e+1:]
              else:
                y=""
                while s<e:
                  g[s]=[l[i] for i,j in enumerate(x) if len(y)<j and j<=len(y+form[s])]
                  y+=form[s]
                  s+=1
                nosp[e]=j[2]
          v=tokenizer(form,add_special_tokens=False)
          w=tokenizer.convert_tokens_to_ids(form)
          i,u=[],[]
          for j,(x,y) in enumerate(zip(v["input_ids"],upos)):
            if j in g and g[j]!=[]:
              x=g[j]
            elif j<len(nosp) and nosp[j] and w[j]!=tokenizer.unk_token_id:
              x=[w[j]]
            if x!=[]:
              i+=x
              u+=[y] if len(x)==1 else ["B-"+y]+["I-"+y]*(len(x)-1)
          if len(i)<tokenizer.model_max_length-3:
            self.ids.append([tokenizer.cls_token_id]+i+[tokenizer.sep_token_id])
            self.upos.append(["SYM"]+u+["SYM"])
          else:
            self.ids.append(i[0:tokenizer.model_max_length-2])
            self.upos.append(u[0:tokenizer.model_max_length-2])
          label=set(sum([self.upos[-1],list(label)],[]))
          form,upos,nosp,mw,mwid,text=[],[],[False],[],[],None
    self.label2id={l:i for i,l in enumerate(sorted(label))}
  def __call__(*args):
    lid={l:i for i,l in enumerate(sorted(set(sum([list(t.label2id) for t in args],[]))))}
    for t in args:
      t.label2id=lid
    return lid
  __len__=lambda self:len(self.ids)
  __getitem__=lambda self,i:{"input_ids":self.ids[i],"labels":[self.label2id[t] for t in self.upos[i]]}
  getpos=lambda self,w:w[3]

class UPOSFeatsDataset(UPOSDataset):
  getpos=lambda self,w:w[3] if w[5]=="_" else w[3]+"|"+w[5]

class UPOSFileDataset(object):
  def __init__(self,conllu,tokenizer):
    self.conllu=open(conllu,"r",encoding="utf-8")
    self.tokenizer=tokenizer
    self.seeks=[0]
    self.multiword={}
    label=set(["SYM"])
    s=self.conllu.readline()
    while s!="":
      if s=="\n":
        self.seeks.append(self.conllu.tell())
      else:
        w=s.split("\t")
        if len(w)==10:
          if w[0].isdecimal():
            label.add(w[3] if w[5]=="_" else w[3]+"|"+w[5])
          elif w[0].find("-")>0:
            t=w[0].split("-")
            f,j,k=w[1],[],[]
            for i in range(int(t[0]),int(t[1])+1):
              w=self.conllu.readline().split("\t")
              j.append(w[3] if w[5]=="_" else w[3]+"|"+w[5])
              k.append(w[1])
            p="+".join(j)
            label.add(p)
            if p in self.multiword:
              self.multiword[p][f]=list(k)
            else:
              self.multiword[p]={f:list(k)}
      s=self.conllu.readline()
    lid={}
    for i,l in enumerate(sorted(label)):
      lid[l],lid["B-"+l],lid["I-"+l]=i*3,i*3+1,i*3+2
    self.label2id=lid
  def __call__(*args):
    lid={l:i for i,l in enumerate(sorted(set(sum([list(t.label2id) for t in args],[]))))}
    for t in args:
      t.label2id=lid
    return lid
  def __del__(self):
    self.conllu.close()
  __len__=lambda self:len(self.seeks)-1
  def __getitem__(self,i):
    self.conllu.seek(self.seeks[i])
    form,upos=[],[]
    while self.conllu.tell()<self.seeks[i+1]:
      w=self.conllu.readline().split("\t")
      if len(w)==10:
        form.append(w[1])
        if w[0].isdecimal():
          upos.append(w[3] if w[5]=="_" else w[3]+"|"+w[5])
        elif w[0].find("-")>0:
          t=w[0].split("-")
          u=[]
          for j in range(int(t[0]),int(t[1])+1):
            k=self.conllu.readline().split("\t")
            u.append(k[3] if k[5]=="_" else k[3]+"|"+k[5])
          upos.append("+".join(u))
    v=self.tokenizer(form,add_special_tokens=False)
    i,u=[],[]
    for j,(x,y) in enumerate(zip(v["input_ids"],upos)):
      if x!=[]:
        i+=x
        u+=[y] if len(x)==1 else ["B-"+y]+["I-"+y]*(len(x)-1)
    if len(i)<self.tokenizer.model_max_length-3:
      ids=[self.tokenizer.cls_token_id]+i+[self.tokenizer.sep_token_id]
      upos=["SYM"]+u+["SYM"]
    else:
      ids=i[0:self.tokenizer.model_max_length-2]
      upos=u[0:self.tokenizer.model_max_length-2]
    return {"input_ids":ids,"labels":[self.label2id[t] for t in upos]}

def makeupos(tmpdir,batch):
  import glob
  if os.path.isdir(sys.argv[3]):
    g=glob.glob(os.path.join(sys.argv[3],"*.conllu"))
  else:
    subprocess.check_output(["git","clone","--depth=1",sys.argv[3]],cwd=tmpdir)
    g=glob.glob(os.path.join(tmpdir,os.path.basename(sys.argv[3]),"*.conllu"))
  if len(g)==1:
    train_file=dev_file=test_file=g[0]
    subprocess.check_output([sys.executable,"-m","esupar.train",sys.argv[1],sys.argv[2],str(batch),tmpdir,train_file])
  elif len(g)==2:
    t=g[0].endswith("train.conllu")
    train_file=g[0] if t else g[1]
    dev_file=test_file=g[1] if t else g[0]
    subprocess.check_output([sys.executable,"-m","esupar.train",sys.argv[1],sys.argv[2],str(batch),tmpdir,train_file,dev_file])
  else:
    g.sort()
    assert g[2].endswith("train.conllu")
    dev_file=g[0]
    test_file=g[1]
    train_file=g[2]
    subprocess.check_output([sys.executable,"-m","esupar.train",sys.argv[1],sys.argv[2],str(batch),tmpdir,train_file,dev_file,test_file])
  return train_file,dev_file,test_file

def trainer(uposdataset):
  from transformers import AutoTokenizer,AutoModelForTokenClassification,AutoConfig,DataCollatorForTokenClassification,TrainingArguments,Trainer
  tokenizer=AutoTokenizer.from_pretrained(sys.argv[1])
  if len(sys.argv)==6:
    train_dts=uposdataset(sys.argv[5],tokenizer)
    eval_dts=None
    label2id=train_dts.label2id
  elif len(sys.argv)==7:
    train_dts=uposdataset(sys.argv[5],tokenizer)
    eval_dts=uposdataset(sys.argv[6],tokenizer)
    label2id=train_dts(eval_dts)
  else:
    train_dts=uposdataset(sys.argv[5],tokenizer)
    eval_dts=uposdataset(sys.argv[6],tokenizer)
    test_dts=uposdataset(sys.argv[7],tokenizer)
    label2id=train_dts(eval_dts,test_dts)
  config=AutoConfig.from_pretrained(sys.argv[1],num_labels=len(label2id),label2id=label2id,id2label={i:l for l,i in label2id.items()})
  if train_dts.multiword!={}:
    if config.task_specific_params:
      config.task_specific_params["upos_multiword"]=train_dts.multiword
    else:
      config.task_specific_params={"upos_multiword":train_dts.multiword}
  model=AutoModelForTokenClassification.from_pretrained(sys.argv[1],config=config)
  arg=TrainingArguments(per_device_train_batch_size=abs(int(sys.argv[3])),output_dir=sys.argv[4],overwrite_output_dir=True,save_total_limit=2,save_strategy="epoch",evaluation_strategy="epoch" if eval_dts else "no")
  train=Trainer(model=model,args=arg,train_dataset=train_dts,eval_dataset=eval_dts,data_collator=DataCollatorForTokenClassification(tokenizer))
  train.train()
  train.save_model(sys.argv[2])
  tokenizer.save_pretrained(sys.argv[2])

if __name__=="__main__":
  batch=32
  if len(sys.argv)==5 and sys.argv[4].startswith("batch="):
    batch=abs(int(sys.argv[4][6:]))
    sys.argv.pop()
  elif len(sys.argv)==3:
    sys.argv.append(".")
  if len(sys.argv)==4:
    with tempfile.TemporaryDirectory() as d:
      a,b,c=makeupos(d,batch)
      subprocess.check_output([sys.executable,"-m","esupar.train",sys.argv[2],sys.argv[2],str(batch),"///",a,b,c])
  elif len(sys.argv)==8 and sys.argv[4]=="///":
    import torch
    from transformers import AutoTokenizer
    p=["biaffine-dep","train","-c","biaffine-dep-en","-b"]
    if torch.cuda.is_available():
      p+=["-d","0"]
    tokenizer=AutoTokenizer.from_pretrained(sys.argv[1])
    p+=["-p",os.path.join(sys.argv[2],"supar.model"),"-f","bert","--bert",sys.argv[1],"--embed=","--unk",tokenizer.unk_token,"--buckets",sys.argv[3],"--train",sys.argv[5],"--dev",sys.argv[6],"--test",sys.argv[7]]
    subprocess.check_output(p)
  elif len(sys.argv)>5 and sys.argv[3].isdecimal():
    trainer(UPOSDataset)
  elif len(sys.argv)>5 and sys.argv[3].startswith("-"):
    trainer(UPOSFeatsDataset)
  elif len(sys.argv)>5 and sys.argv[3].startswith("+"):
    trainer(UPOSFileDataset)
  else:
    print("Usage:",os.path.basename(sys.executable),"-m esupar.train source-model target-model UD_URL [batch=32]")

