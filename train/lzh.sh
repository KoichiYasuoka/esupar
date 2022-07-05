#! /bin/sh
pip3 install -U esupar pytokenizations
test -d UD_Classical_Chinese-Kyoto || git clone --depth=1 https://github.com/UniversalDependencies/UD_Classical_Chinese-Kyoto
for F in train dev test
do cat UD_Classical_Chinese-Kyoto/*-$F*.conllu > $F.conllu
done
python3 -m esupar.train KoichiYasuoka/roberta-classical-chinese-base-char KoichiYasuoka/roberta-classical-chinese-base-upos .
python3 -m esupar.train KoichiYasuoka/roberta-classical-chinese-large-char KoichiYasuoka/roberta-classical-chinese-large-upos . batch=8
python3 -m esupar.train Jihuai/bert-ancient-chinese KoichiYasuoka/bert-ancient-chinese-base-upos .
exit 0
