#! /bin/sh
pip3 install -U esupar fugashi unidic-lite pytokenizations
test -d UD_Japanese-GSDLUW || git clone --depth=1 https://github.com/UniversalDependencies/UD_Japanese-GSDLUW

python3 -m esupar.train KoichiYasuoka/roberta-base-japanese-aozora KoichiYasuoka/roberta-base-japanese-luw-upos UD_Japanese-GSDLUW
python3 -m esupar.train KoichiYasuoka/roberta-base-japanese-aozora-char KoichiYasuoka/roberta-base-japanese-char-luw-upos UD_Japanese-GSDLUW
python3 -m esupar.train KoichiYasuoka/roberta-large-japanese-aozora KoichiYasuoka/roberta-large-japanese-luw-upos UD_Japanese-GSDLUW batch=3
python3 -m esupar.train KoichiYasuoka/roberta-large-japanese-aozora-char KoichiYasuoka/roberta-large-japanese-char-luw-upos UD_Japanese-GSDLUW batch=8

python3 -m esupar.train cl-tohoku/bert-base-japanese-v2 KoichiYasuoka/bert-base-japanese-unidic-luw-upos UD_Japanese-GSDLUW
python3 -m esupar.train KoichiYasuoka/bert-base-japanese-char-extended KoichiYasuoka/bert-base-japanese-luw-upos UD_Japanese-GSDLUW
python3 -m esupar.train cl-tohoku/bert-large-japanese KoichiYasuoka/bert-large-japanese-unidic-luw-upos UD_Japanese-GSDLUW batch=8
python3 -m esupar.train KoichiYasuoka/bert-large-japanese-char-extended KoichiYasuoka/bert-large-japanese-luw-upos UD_Japanese-GSDLUW batch=8

exit 0
