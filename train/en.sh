#! /bin/sh
pip3 install -U esupar pytokenizations
for U in UD_English-EWT UD_English-GUM UD_English-ParTUT UD_English-Lines UD_English-Atis
do test -d $U || git clone --depth=1 https://github.com/UniversalDependencies/$U
done
for F in train dev test
do cat UD_English-*/*-$F.conllu > $F.conllu
done
python3 -m esupar.train roberta-base KoichiYasuoka/roberta-base-english-upos . batch=16
python3 -m esupar.train roberta-large KoichiYasuoka/roberta-large-english-upos . batch=4
exit 0
