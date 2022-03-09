#! /bin/sh
pip3 install -U esupar pytokenizations
if [ -d spaCy-Thai ]
then ( cd spaCy-Thai && git pull )
else git clone --depth=1 https://github.com/KoichiYasuoka/spaCy-Thai
fi
s='{print>"train.upos";if(NF>0)u=u$0"\n";else{f=FILENAME;if(u~/\t0\troot\t/)print u>(f~/-dev/?"dev":f~/-test/?"test":f~/-train/?"train":NR%10<1?"dev":NR%10<2?"test":"train")".conllu";u=""}}'
nawk -F'\t' "$s" spaCy-Thai/UD_Thai-Corpora/*-ud-*.conllu

python3 -m esupar.train KoichiYasuoka/roberta-base-thai-char KoichiYasuoka/roberta-base-thai-char-upos 32 /tmp train.upos
python3 -m esupar.train KoichiYasuoka/roberta-base-thai-char-upos KoichiYasuoka/roberta-base-thai-char-upos 32 /// train.conllu dev.conllu test.conllu
python3 -m esupar.train KoichiYasuoka/roberta-base-thai-spm KoichiYasuoka/roberta-base-thai-spm-upos 32 /tmp train.upos
python3 -m esupar.train KoichiYasuoka/roberta-base-thai-spm-upos KoichiYasuoka/roberta-base-thai-spm-upos 32 /// train.conllu dev.conllu test.conllu
python3 -m esupar.train KoichiYasuoka/roberta-base-thai-syllable KoichiYasuoka/roberta-base-thai-syllable-upos 32 /tmp train.upos
python3 -m esupar.train KoichiYasuoka/roberta-base-thai-syllable-upos KoichiYasuoka/roberta-base-thai-syllable-upos 32 /// train.conllu dev.conllu test.conllu

exit 0
