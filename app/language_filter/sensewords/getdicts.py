#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @time: 2022/10/11 
# @Author: czx
import os.path

import jieba
import collections
from app.language_filter.uniform.utils import uniform, remove_punctuation, chinese_string_len
import json

# 词典分为词、短语（句子、网址）
# 统一中英文单词词典
# todo: 网址识别，网址词典。短语是否作为自定义词典


def is_token(x, english=False):
    if english:
        if len(x.split()) > 1:
            return False
        elif len(x.split()) == 1:
            return True
        else:
            # 空
            return False
    # chinese
    else:
        if len(x) <= 3:
            return True
        elif 1 <= chinese_string_len(x) <= 2 and len(x) < 10:
            return True
        return False


def load_dict(dict_file, english=False):
    words = open(dict_file, 'r').readlines()
    # words
    words = [x.strip('\n').split('\t')[0] for x in words]

    #
    if english:
        word_dict = set([uniform(x) for x in words])
    else:
        word_dict = set([remove_punctuation(uniform(x)) for x in words])
    return word_dict


def get_dict_words(input_file, dict_file, english=False):
    #
    if os.path.exists(dict_file):
        word_dict = load_dict(dict_file)
    else:
        word_dict = set()

    encoding = 'utf-16' if "涉枪涉爆违法信息关键词.txt" in input_file else 'utf-8'

    with open(input_file, 'r', encoding=encoding) as f, open(dict_file, 'a+') as wf:
        if input_file.endswith(".json"):
            # 对words.json的处理
            f = json.load(f)

        for x in f:
            # 大小写转换，全角半角转换
            x = uniform(x).strip('\n')

            # 英文单词词典、短语词典
            if english:
                if not is_token(x, english):
                    continue

                if x not in word_dict:
                    wf.write(x+'\n')
            else:
                # 去除标点符号
                x = remove_punctuation(x)
                if is_token(x) and x not in word_dict:
                    wf.write(x+'\t'+str(500)+'\n')


    # 长的句子或短语：
    #   做分词。分词后，分词后本身包含在词典中，则不必保留。分词后，没有一个词在词典中，作为一个整体。
    #   做分词，分词后。如果已经在词典中，则不做词频统计。否则做词频统计。出现次数多的，放入词典中。


def token_in_dict(tokens, word_dict):
    for x in tokens:
        if x in word_dict:
            return True
    return False


def get_dict_phrase(input_file, word_file, phrase_file, new_word_file, english=False):
    # word_dict
    word_dict = load_dict(word_file)
    jieba.load_userdict(word_file)
    new_word_dict = {}

    # phrase_dict

    if os.path.exists(phrase_file):
        phrase = open(phrase_file, 'r').readlines()
        phrase_dict = set([remove_punctuation(uniform(x).strip('\n')) for x in phrase])
    else:
        phrase_dict = set()

    #
    encoding = 'utf-16' if "涉枪涉爆违法信息关键词.txt" in input_file else 'utf-8'
    with open(input_file, 'r', encoding=encoding) as f, open(phrase_file, 'a+') as wf:
        if input_file.endswith(".json"):
            # 对words.json的处理
            f = json.load(f)

        for x in f:
            # uniform
            x = uniform(x).strip('\n')

            # 去除标点符号
            if not english:
                x = remove_punctuation(x)

            # token不做处理
            if is_token(x, english):
                continue

            tokens = [x[0] for x in list(jieba.tokenize(x))] if not english else x.split()
            if token_in_dict(tokens, word_dict):
                continue

            for k in tokens:
                if k not in new_word_dict:
                    new_word_dict[k] = 1
                else:
                    new_word_dict[k] += 1

            if x not in phrase_dict:
                wf.write(x+'\n')

    if new_word_file:
        with open(new_word_file, 'w') as wf:
            for word in new_word_dict:
                wf.write(json.dumps({word: new_word_dict[word]}, ensure_ascii=False)+'\n')


if __name__ == "__main__":
    input_dir = "../english"
    files = os.listdir(input_dir)

    word_dict_file = "dicts/english_dict.txt"
    new_word_dict = "dicts/phrase_dict.txt"
    phrase_dict_file = "dicts/chinese_phrase.txt"

    for x in files:
        if x == "网址.txt":
            continue
        input_file = os.path.join(input_dir, x)
        print('file is:', x)
        get_dict_words(input_file, word_dict_file, english=True)
        # get_dict_phrase(input_file, word_dict_file, phrase_dict_file, new_word_dict)