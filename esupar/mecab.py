#! /usr/bin/python -i
# coding=utf-8

from transformers import BertTokenizerFast
from transformers.models.bert_japanese.tokenization_bert_japanese import MecabTokenizer

class MecabPreTokenizer(MecabTokenizer):
  def mecab_split(self,i,normalized_string):
    t=str(normalized_string)
    e=0
    z=[]
    for c in self.tokenize(t):
      s=t.find(c,e)
      e=e if s<0 else s+len(c)
      z.append((0,0) if s<0 else (s,e))
    return [normalized_string[s:e] for s,e in z if e>0]
  def pre_tokenize(self,pretok):
    pretok.split(self.mecab_split)

class BertMecabTokenizerFast(BertTokenizerFast):
  def __init__(self,vocab_file,do_lower_case=False,tokenize_chinese_chars=False,**kwargs):
    from tokenizers import pre_tokenizers,normalizers
    super().__init__(vocab_file=vocab_file,do_lower_case=do_lower_case,tokenize_chinese_chars=tokenize_chinese_chars,**kwargs)
    d=kwargs["mecab_kwargs"] if "mecab_kwargs" in kwargs else {"mecab_dic":"ipadic"}
    self._tokenizer.normalizer=normalizers.Sequence([normalizers.Nmt(),normalizers.NFKC()])
    self._tokenizer.pre_tokenizer=pre_tokenizers.Sequence([pre_tokenizers.PreTokenizer.custom(MecabPreTokenizer(**d)),pre_tokenizers.BertPreTokenizer()])

