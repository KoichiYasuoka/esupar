#! /bin/sh
pip3 install -U esupar spacy-alignments
test -d UD_Korean-Kaist || git clone --depth=1 https://github.com/UniversalDependencies/UD_Korean-Kaist
test -d UD_Korean-GSD || git clone --depth=1 https://github.com/UniversalDependencies/UD_Korean-GSD
for F in train dev test
do nawk '
BEGIN{
  FS=OFS="\t";
}
{
  if(NF==10){
    if($4=="ADP"&&$8~/^(nsubj|obj)$/)
      $8="case";
    else if($2~/^않/&&$4=="VERB"&&$8=="flat"){
      $4="AUX";
      $8="aux";
    }
  }
  print;
}' UD_Korean-*/*-$F.conllu > $F.conllu
done
python3 -m esupar.train KoichiYasuoka/roberta-base-korean-hanja KoichiYasuoka/roberta-base-korean-upos .
python3 -m esupar.train KoichiYasuoka/roberta-large-korean-hanja KoichiYasuoka/roberta-large-korean-upos .
exit 0
