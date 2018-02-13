import os
import re


def get_word(line):
    return line.replace("-", '') + '\t'


def get_morphemes(line):
    first = True
    text = ''
    morph = line.split('\t')
    for m in morph:
        if first:
            first=False
            text +=m+'\t'
        else:
            text +=' '+m + ':' + m
    return text.lower()

file = open('train.txt', 'r')
text = file.read().split('\n')
file.close()

file = open('clean_train.txt', 'w')
for line in text:
    file.write(get_morphemes(line) + '\n')

file.close()
