#! /usr/bin/python -i
# coding=utf-8

import os,sys,subprocess,tempfile

class UPOSDataset(object):
  def __init__(self,conllu,tokenizer):
    self.ids=[]
    self.upos=[]
    with open(conllu,"r",encoding="utf-8") as f:
      i,u=[],[]
      for t in f:
        w=t.split("\t")
        if len(w)==10 and w[0].isdecimal():
          v=tokenizer(w[1],add_special_tokens=False)["input_ids"]
          i+=v
          u+=[w[3]] if len(v)==1 else ["B-"+w[3]]+["I-"+w[3]]*(len(v)-1)
        elif t.strip()=="" and len(i)>0:
          self.ids.append([tokenizer.cls_token_id]+i+[tokenizer.sep_token_id])
          self.upos.append(["SYM"]+u+["SYM"])
          i,u=[],[]
    self.label2id={l:i for i,l in enumerate(sorted(set(sum(self.upos,[]))))}
  def __call__(*args):
    lid={l:i for i,l in enumerate(sorted(set(sum([list(t.label2id) for t in args],[]))))}
    for t in args:
      t.label2id=lid
    return lid
  __len__=lambda self:len(self.ids)
  __getitem__=lambda self,i:{"input_ids":self.ids[i],"labels":[self.label2id[t] for t in self.upos[i]]}

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

def trainer():
  from transformers import AutoTokenizer,AutoModelForTokenClassification,AutoConfig,DataCollatorForTokenClassification,TrainingArguments,Trainer
  tokenizer=AutoTokenizer.from_pretrained(sys.argv[1])
  if len(sys.argv)==6:
    train_dts=UPOSDataset(sys.argv[5],tokenizer)
    eval_dts=None
    label2id=train_dts.label2id
  elif len(sys.argv)==7:
    train_dts=UPOSDataset(sys.argv[5],tokenizer)
    eval_dts=UPOSDataset(sys.argv[6],tokenizer)
    label2id=train_dts(eval_dts)
  else:
    train_dts=UPOSDataset(sys.argv[5],tokenizer)
    eval_dts=UPOSDataset(sys.argv[6],tokenizer)
    test_dts=UPOSDataset(sys.argv[7],tokenizer)
    label2id=train_dts(eval_dts,test_dts)
  config=AutoConfig.from_pretrained(sys.argv[1],num_labels=len(label2id),label2id=label2id,id2label={i:l for l,i in label2id.items()})
  model=AutoModelForTokenClassification.from_pretrained(sys.argv[1],config=config)
  arg=TrainingArguments(per_device_train_batch_size=int(sys.argv[3]),output_dir=sys.argv[4],overwrite_output_dir=True,save_total_limit=2,save_strategy="epoch",evaluation_strategy="epoch" if eval_dts else "no")
  train=Trainer(model=model,args=arg,train_dataset=train_dts,eval_dataset=eval_dts,data_collator=DataCollatorForTokenClassification(tokenizer))
  train.train()
  train.save_model(sys.argv[2])
  tokenizer.save_pretrained(sys.argv[2])

if __name__=="__main__":
  batch=32
  if len(sys.argv)==5 and sys.argv[4].startswith("batch="):
    batch=int(sys.argv[4][6:])
    sys.argv.pop()
  if len(sys.argv)==4:
    with tempfile.TemporaryDirectory() as d:
      import torch
      from transformers import AutoTokenizer
      a,b,c=makeupos(d,batch)
      p=["biaffine-dep","train","-c","biaffine-dep-en","-b"]
      if torch.cuda.is_available():
        p+=["-d","0"]
        torch.cuda.empty_cache()
      tokenizer=AutoTokenizer.from_pretrained(sys.argv[2])
      p+=["-p",os.path.join(sys.argv[2],"supar.model"),"-f","bert","--bert",sys.argv[2],"--embed=","--unk",tokenizer.unk_token,"--train",a,"--dev",b,"--test",c]
      subprocess.check_output(p)
  elif len(sys.argv)>5 and sys.argv[3].isdecimal():
    trainer()
  else:
    print("Usage:",os.path.basename(sys.executable),"-m esupar.train source-model target-model UD_URL [batch=32]")

