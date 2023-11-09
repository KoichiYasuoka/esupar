#! /bin/sh
pip3 install -U esupar spacy-alignments
test -d morphUD-korean || git clone --depth=1 https://github.com/jungyeul/morphUD-korean
for F in train dev test
do nawk '
BEGIN{
  FS=OFS="\t";
}
{
  if(NF==10){
    if($8=="aux"){
      if($4=="PART"){
        if($5=="XSN")
          $8="fixed";
        else if($5!="XSV")
          $8="mark";
      }
      else if($4=="CCONJ")
        $8=($5=="JC")?"cc":"mark";
      else if($5=="VCP")
        $8="cop";
    }
    else if($4=="ADP"){
      if($8~/^(nsubj|obj)$/)
        $8="case";
    }
  }
  print;
}' morphUD-korean/data/ko_gsd-ud-$F-morph.conllu > $F.conllu
done
python3 -m esupar.train KoichiYasuoka/roberta-base-korean-upos KoichiYasuoka/roberta-base-korean-morph-upos .
python3 -m esupar.train KoichiYasuoka/roberta-large-korean-upos KoichiYasuoka/roberta-large-korean-morph-upos .
python3 -m esupar.train KoichiYasuoka/deberta-base-korean-upos KoichiYasuoka/deberta-base-korean-morph-upos .
exit 0
