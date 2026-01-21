#! /bin/sh
pip3 install -U esupar spacy-alignments
for U in UD_English-EWT UD_English-GUM UD_English-ParTUT UD_English-Lines UD_English-Atis
do test -d $U || git clone --depth=1 https://github.com/UniversalDependencies/$U
done
for F in train dev test
do cat UD_English-*/*-$F.conllu > $F.conllu
done
python3 -m esupar.train FacebookAI/roberta-base KoichiYasuoka/roberta-base-english-upos -32 /tmp train.conllu
python3 -m esupar.train KoichiYasuoka/roberta-base-english-upos KoichiYasuoka/roberta-base-english-upos 32 /// train.conllu dev.conllu test.conllu
python3 -m esupar.train FacebookAI/roberta-large KoichiYasuoka/roberta-large-english-upos -8 /tmp train.conllu
python3 -m esupar.train KoichiYasuoka/roberta-large-english-upos KoichiYasuoka/roberta-large-english-upos 8 /// train.conllu dev.conllu test.conllu
exit 0
