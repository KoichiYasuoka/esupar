#! /bin/sh
pip3 install -U esupar pytokenizations
test -d morphUD-korean || git clone --depth=1 https://github.com/jungyeul/morphUD-korean
for F in train dev test
do cp morphUD-korean/data/ko_gsd-ud-$F-morph.conllu $F.conllu
done
python3 -m esupar.train klue/roberta-base KoichiYasuoka/roberta-base-korean-morph-upos .
python3 -m esupar.train klue/roberta-large KoichiYasuoka/roberta-large-korean-morph-upos .
exit 0
