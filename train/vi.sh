#! /bin/sh
pip3 install -U esupar pytokenizations
test -d UD_Vietnamese-VTB || git clone --depth=1 https://github.com/UniversalDependencies/UD_Vietnamese-VTB
python3 -c '
from transformers import BertTokenizer,AutoModelForMaskedLM
mdl=AutoModelForMaskedLM.from_pretrained("FPTAI/vibert-base-cased")
tkz=BertTokenizer.from_pretrained("FPTAI/vibert-base-cased",do_lower_case=False,strip_accents=False,model_max_length=mdl.config.max_position_embeddings)
tkz.save_pretrained("tmpdir")
mdl.save_pretrained("tmpdir")
'
python3 -m esupar.train tmpdir KoichiYasuoka/bert-base-vietnamese-upos UD_Vietnamese-VTB
exit 0
