[![PyPI - PyPi](https://img.shields.io/pypi/v/lingtrain-aligner)](https://pypi.org/project/lingtrain-aligner) [![Downloads](https://static.pepy.tech/personalized-badge/lingtrain-aligner?period=total&units=abbreviation&left_color=grey&right_color=green&left_text=Downloads)](https://pepy.tech/project/lingtrain-aligner)

# Lingtrain Aligner

ML powered library for the accurate texts alignment in different languages.

- 🔥 [How to create bilingual books. Part 2. Lingtrain Alignment Studio](https://habr.com/ru/post/590549/)
- 💡 [Lingtrain Aligner. How to make parallel books for language learning. Part 1. Python and Colab version](https://habr.com/ru/post/586574/)

![Cover](https://i.imgur.com/WQWB4v0.png)

## Purpose

Main purpose of this alignment tool is to build parallel corpora using two or more raw texts in different languages. Texts should contain the same information (i.e., one text should be a translated analog oh the other text). E.g., it can be the _Drei Kameraden_ by Remarque in German and the _Three Comrades_ — it's translation into English.

## Process

There are plenty of obstacles during the alignment process:

- The translator could translate several sentences as one.
- The translator could translate one sentence as many.
- There are some service marks in the text
    - Page numbers
    - Chapters and other section headings
    - Author and title information
    - Notes

While service marks can be handled manually (the tool helps to detect them), the translation conflicts should be handled more carefully.

Lingtrain Aligner tool will do almost all alignment work for you. It matches the sentence pairs automatically using the multilingual machine learning models. Then it searches for the alignment conflicts and resolves them. As output you will have the parallel corpora either as two distinct plain text files or as the merged corpora in widely used TMX format.

### Supported languages and models

Automated alignment process relies on the sentence embeddings models. Embeddings are multidimensional vectors of a special kind which are used to calculate a distance between the sentences. Supported languages list depend on the selected backend model.

- **distiluse-base-multilingual-cased-v2**
  - more reliable and fast
  - moderate weights size — 500MB
  - supports 50+ languages
  - full list of supported languages can be found in [this paper](https://arxiv.org/abs/2004.09813)
- **LaBSE (Language-agnostic BERT Sentence Embedding)**
  - can be used for rare languages
  - pretty heavy weights — 1.8GB
  - supports 100+ languages
  - full list of supported languages can be found [here](https://arxiv.org/abs/2007.01852)
- **SONAR** (Sentence-level multimOdal and laNguage-Agnostic Representations)
  - Supports about 200 languages (approximately [these](https://github.com/facebookresearch/flores/tree/main/flores200))
  - A large model (3 GB of weights)
  - Ideally, requires you to indicate the source language explicitly
  - Was originally released at [facebookresearch/SONAR](https://github.com/facebookresearch/SONAR) based on [fairseq2](https://github.com/facebookresearch/fairseq2), 
  but here uses [a HuggingFace port](https://huggingface.co/cointegrated/SONAR_200_text_encoder).
  

## Profit

- Parallel corpora by itself can used as the resource for machine translation models or for linguistic researches.
- My personal goal of this project is to help people building parallel translated books for the foreign language learning. 


