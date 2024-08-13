#! /bin/sh
test -d UD_Belarusian-HSE || git clone --depth=1 https://github.com/UniversalDependencies/UD_Belarusian-HSE
for F in train dev test
do cp UD_Belarusian-HSE/*-$F.conllu $F.conllu
done
cat *.conllu > train.upos
python3 -m esupar.train KoichiYasuoka/roberta-small-belarusian KoichiYasuoka/roberta-small-belarusian-upos -32 /tmp train.upos
python3 -m esupar.train KoichiYasuoka/roberta-small-belarusian-upos KoichiYasuoka/roberta-small-belarusian-upos 32 /// train.conllu dev.conllu test.conllu
python3 -m esupar.train KoichiYasuoka/deberta-base-belarusian KoichiYasuoka/deberta-base-belarusian-upos -32 /tmp train.upos
python3 -m esupar.train KoichiYasuoka/deberta-base-belarusian-upos KoichiYasuoka/deberta-base-belarusian-upos 32 /// train.conllu dev.conllu test.conllu
