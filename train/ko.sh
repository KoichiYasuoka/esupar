#! /bin/sh
pip3 install -U esupar pytokenizations
test -d UD_Korean-Kaist || git clone --depth=1 https://github.com/UniversalDependencies/UD_Korean-Kaist
test -d UD_Korean-GSD || git clone --depth=1 https://github.com/UniversalDependencies/UD_Korean-GSD
for F in train dev test
do cat UD_Korean-*/*-$F.conllu > $F.conllu
done
python3 -m esupar.train klue/roberta-base KoichiYasuoka/roberta-base-korean-upos .
python3 -m esupar.train klue/roberta-large KoichiYasuoka/roberta-large-korean-upos .
exit 0
