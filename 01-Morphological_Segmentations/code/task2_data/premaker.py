import os
import re


def get_word(line):
    return line.replace("-", '') + '\t'


def get_morphemes(line):
    text = ''
    morph = line.split('-')
    for m in morph:
        text += m + ':' + m + ' '
    return text

file = open('dev.ita.txt', 'r')
text = file.read().split()
file.close()

file = open('dev.it.txt', 'w')
for line in text:
    file.write(get_word(line) + get_morphemes(line) + '\n')

file.close()
