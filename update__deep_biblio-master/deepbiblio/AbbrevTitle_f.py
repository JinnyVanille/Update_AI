# author: Tang Tiong Yew
# email: tiongyewt@sunway.edu.my
# Project Title: Deep Bilio: A Python Tool for Deep Learning Biliometric Analysis
# Copyright 2023
#
def AbbrevTitle(title=''):
    if title == '':
        return ''
    words = title.split()
    stopwords = ['AND', 'THE', 'OF', 'IN', 'FOR', 'TO', 'AN', 'A', 'BY', 'WITH', 'ON', 'AT']
    abbrev = ''.join([word[0] for word in words if word.upper() not in stopwords])
    return abbrev