# Lingtrain Aligner

ML powered library for the accurate texts alignment in different languages.

## Purpose

Main purpose of this alignment tool is to build parallel corpora using two or more raw texts in different languages. Texts should contain the same information (i.e., one text should be a translated analog oh the other text). E.g., it can be the _Drei Kameraden_ by Remqrque in German and the _Three Comrades_ — it's translation into English.

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

## Profit

- Parallel corpora by itself can used as the resource for machine translation models or for linguistic researches.
- My personal goal of this project is to help people building parallel translated books for the foreign language learning. 
