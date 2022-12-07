#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @time: 2022/10/11 
# @Author: czx
import jieba
from app.language_filter.uniform.utils import uniform, remove_punctuation, replace_punctuation
from .getdicts import load_dict


class SenseFilter:
    def __init__(self, word_dict, phrase_dict=None, english=False):
        # 中文一个词典
        # 英文包括phrase 词典
        self.word_dict = load_dict(word_dict)
        self.english = english

        self.phrase_dict = load_dict(phrase_dict, english) if english else {}

        if not self.english:
            jieba.load_userdict(word_dict)

    def get_tokens(self, sent):
        if self.english:
            sent = replace_punctuation(sent)
            tokens = sent.split()
        else:
            tokens = [x[0] for x in list(jieba.tokenize(sent))]
        return tokens

    def filter(self, sent):
        """
        Returns:
        """
        # 对句子做归一化处理
        sent = uniform(sent)

        if not sent:
            return {"0": ""}

        if not self.english:
            sent = remove_punctuation(sent)

        # 短语判别
        if sent in self.phrase_dict or (len(sent) > 1 and sent in self.word_dict):
            return {'1': sent}

        # for english 短语判别
        if self.phrase_dict:
            for x in self.phrase_dict:
                if x in sent:
                    return {'1': x}

        # 词语判别
        tokens = self.get_tokens(sent)
        # 判断词语
        for x in tokens:
            if len(x) > 1 and x in self.word_dict:
                return {"1": x}

        return {"0": ""}