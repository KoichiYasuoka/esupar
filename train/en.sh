#! /bin/sh
pip3 install -U esupar pytokenizations
python3 -m esupar.train roberta-base KoichiYasuoka/roberta-base-english-upos https://github.com/UniversalDependencies/UD_English-EWT
python3 -m esupar.train xlm-roberta-base KoichiYasuoka/xlm-roberta-base-english-upos https://github.com/UniversalDependencies/UD_English-EWT batch=8
exit 0
