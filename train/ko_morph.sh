#! /bin/sh
pip3 install -U esupar pytokenizations
test -d morphUD-korean || git clone --depth=1 https://github.com/jungyeul/morphUD-korean
for F in train dev test
do nawk '
BEGIN{
  IFS=OFS="\t";
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
  }
  print;
}' morphUD-korean/data/ko_gsd-ud-$F-morph.conllu > $F.conllu
done
python3 -m esupar.train KoichiYasuoka/roberta-base-korean-hanja KoichiYasuoka/roberta-base-korean-morph-upos .
python3 -m esupar.train KoichiYasuoka/roberta-large-korean-hanja KoichiYasuoka/roberta-large-korean-morph-upos .
exit 0
