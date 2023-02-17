import os,setuptools
with open("README.md","r",encoding="utf-8") as r:
  long_description=r.read()
URL="https://github.com/KoichiYasuoka/esupar"

setuptools.setup(
  name="esupar",
  version="1.6.1",
  description="Tokenizer POS-tagger and Dependency-parser with BERT/RoBERTa/DeBERTa models for Japanese and other languages",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url=URL,
  author="Koichi Yasuoka",
  author_email="yasuoka@kanji.zinbun.kyoto-u.ac.jp",
  license="MIT",
  keywords="NLP Japanese Korean Chinese Thai Vietnamese English German Serbian Coptic Ainu",
  packages=setuptools.find_packages(),
  install_requires=[
    "supar>=1.1.4",
    "transformers>=4.19.0",
    "deplacy>=2.0.5"
  ],
  python_requires=">=3.7",
  classifiers=[
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Operating System :: OS Independent",
    "Natural Language :: Japanese",
    "Natural Language :: Korean",
    "Natural Language :: Chinese (Simplified)",
    "Natural Language :: Chinese (Traditional)",
    "Natural Language :: Thai",
    "Natural Language :: Vietnamese",
    "Natural Language :: English",
    "Natural Language :: German",
    "Natural Language :: Serbian",
    "Topic :: Text Processing :: Linguistic"
  ],
  project_urls={
    "Source":URL,
    "Tracker":URL+"/issues",
  }
)
