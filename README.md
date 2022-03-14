[![Current PyPI packages](https://badge.fury.io/py/esupar.svg)](https://pypi.org/project/esupar/)

# esupar

Tokenizer, POS-tagger, and dependency-parser with [Transformers](https://huggingface.co/transformers/) and [SuPar](https://pypi.org/project/supar/).

## Basic usage

```py
>>> import esupar
>>> nlp=esupar.load("ja")
>>> doc=nlp("太郎は花子が読んでいる本を次郎に渡した")
>>> print(doc)
1	太郎	_	PROPN	_	_	12	nsubj	_	SpaceAfter=No
2	は	_	ADP	_	_	1	case	_	SpaceAfter=No
3	花子	_	PROPN	_	_	5	nsubj	_	SpaceAfter=No
4	が	_	ADP	_	_	3	case	_	SpaceAfter=No
5	読ん	_	VERB	_	_	8	acl	_	SpaceAfter=No
6	で	_	SCONJ	_	_	5	mark	_	SpaceAfter=No
7	いる	_	AUX	_	_	5	aux	_	SpaceAfter=No
8	本	_	NOUN	_	_	12	obj	_	SpaceAfter=No
9	を	_	ADP	_	_	8	case	_	SpaceAfter=No
10	次郎	_	PROPN	_	_	12	obl	_	SpaceAfter=No
11	に	_	ADP	_	_	10	case	_	SpaceAfter=No
12	渡し	_	VERB	_	_	0	root	_	SpaceAfter=No
13	た	_	AUX	_	_	12	aux	_	_

>>> import deplacy
>>> deplacy.render(doc,Japanese=True)
太郎 PROPN ═╗<════════╗ nsubj(主語)
は   ADP   <╝         ║ case(格表示)
花子 PROPN ═╗<══╗     ║ nsubj(主語)
が   ADP   <╝   ║     ║ case(格表示)
読ん VERB  ═╗═╗═╝<╗   ║ acl(連体修飾節)
で   SCONJ <╝ ║   ║   ║ mark(標識)
いる AUX   <══╝   ║   ║ aux(動詞補助成分)
本   NOUN  ═╗═════╝<╗ ║ obj(目的語)
を   ADP   <╝       ║ ║ case(格表示)
次郎 PROPN ═╗<╗     ║ ║ obl(斜格補語)
に   ADP   <╝ ║     ║ ║ case(格表示)
渡し VERB  ═╗═╝═════╝═╝ root(親)
た   AUX   <╝           aux(動詞補助成分)
```

`esupar.load(model)` loads a natural language processor pipeline, working on [Universal Dependencies](https://universaldependencies.org/format.html). Available `model` options are:

* `model="ja"` Japanese model [bert-base-japanese-upos](https://huggingface.co/KoichiYasuoka/bert-base-japanese-upos) (default)
* `model="ja_large"` Japanese model [bert-large-japanese-upos](https://huggingface.co/KoichiYasuoka/bert-large-japanese-upos)
* `model="ja_luw_small"` Japanese long-unit-word model [roberta-small-japanese-char-luw-upos](https://huggingface.co/KoichiYasuoka/roberta-small-japanese-char-luw-upos)
* `model="ja_luw_base"` Japanese long-unit-word model [bert-base-japanese-luw-upos](https://huggingface.co/KoichiYasuoka/bert-base-japanese-luw-upos)
* `model="ja_luw_large"` Japanese long-unit-word model [bert-large-japanese-luw-upos](https://huggingface.co/KoichiYasuoka/bert-large-japanese-luw-upos)
* `model="zh"` Chinese model [chinese-bert-wwm-ext-upos](https://huggingface.co/KoichiYasuoka/chinese-bert-wwm-ext-upos)
* `model="zh_base"` Chinese model [chinese-roberta-base-upos](https://huggingface.co/KoichiYasuoka/chinese-roberta-base-upos)
* `model="zh_large"` Chinese model [chinese-roberta-large-upos](https://huggingface.co/KoichiYasuoka/chinese-roberta-large-upos)
* `model="lzh"` Classical Chinese model [roberta-classical-chinese-base-upos](https://huggingface.co/KoichiYasuoka/roberta-classical-chinese-base-upos)
* `model="lzh_large"` Classical Chinese model [roberta-classical-chinese-large-upos](https://huggingface.co/KoichiYasuoka/roberta-classical-chinese-large-upos)
* `model="th"` Thai model [roberta-base-thai-char-upos](https://huggingface.co/KoichiYasuoka/roberta-base-thai-char-upos)
* `model="en"` English model [roberta-base-english-upos](https://huggingface.co/KoichiYasuoka/roberta-base-english-upos)
* `model="en_large"` English model [roberta-large-english-upos](https://huggingface.co/KoichiYasuoka/roberta-large-english-upos)
* `model="de"` German model [bert-base-german-upos](https://huggingface.co/KoichiYasuoka/bert-base-german-upos)
* `model="de_large"` German model [bert-large-german-upos](https://huggingface.co/KoichiYasuoka/bert-large-german-upos)

## Installation for Linux

```sh
pip3 install esupar --user
```

## Installation for Cygwin64

Make sure to get `python37-devel` `python37-pip` `python37-cython` `python37-numpy` `python37-wheel` `gcc-g++` `mingw64-x86_64-gcc-g++` `git` `curl` `make` `cmake`, and then:

```sh
curl -L https://raw.githubusercontent.com/KoichiYasuoka/CygTorch/master/installer/supar.sh | sh
pip3.7 install esupar
```

## Installation for Google Colaboratory

```py
!pip install esupar
```

Try [notebook](https://colab.research.google.com/github/KoichiYasuoka/esupar/blob/master/esupar.ipynb).

## Author

Koichi Yasuoka (安岡孝一)

