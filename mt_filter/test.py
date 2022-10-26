from filter import DFAFilter
import json
import os

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

detecter = DFAFilter()

ps = PorterStemmer()

data_root = './english/'

def get_words(data_root):
    data_path_list = [data_root + f for f in os.listdir(data_root)]

    ret = []
    for f in data_path_list:
        if f.endswith('json'):
            ret += json.load(open(f))
        else:
            for l in open(f):
                ret += l.strip()

    ret = sorted(list(set(ret)))

    ret = [l for l in ret if len(l)>2]

    print(f'get {len(ret)} bad words')

    return ret

bad_words = get_words(data_root)

for bd in bad_words:
    try:
        detecter.add(bd)
    except:
        print(bd)

inputs = 'sexy girl, fuck you, Trump is an asshole, very very beautiful'
while True:
    print(inputs)
    print('=> ' + detecter.filter(inputs))
    inputs = input()
