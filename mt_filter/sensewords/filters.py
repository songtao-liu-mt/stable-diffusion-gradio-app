#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @time: 2022/10/11 
# @Author: czx
import jieba
from mt_filter.uniform.utils import uniform, remove_punctuation
from .getdicts import load_dict


class SenseFilter:
    def __init__(self, word_dict, phrase_dict, english=False):
        # 初始化词典，把敏感词词典作为特殊词典
        # 敏感词分为两部分，词典和短语。
        self.word_dict = load_dict(word_dict)
        self.phrase_dict = load_dict(phrase_dict)

        self.english = english

        if not self.english:
            jieba.load_userdict(word_dict)

    def get_tokens(self, sent):
        if self.english:
            tokens = sent.split('')
        else:
            tokens = [x[0] for x in list(jieba.tokenize(sent))]
        return tokens

    def filter(self, sent):
        """

        Returns:

        """
        # 对句子做归一化处理
        sent = uniform(sent)

        #
        if not self.english:
            sent = remove_punctuation(sent)

        # 短语判别，网址
        if sent in self.phrase_dict or sent in self.word_dict:
            return {'1': sent}

        # 词语判别
        tokens = self.get_tokens(sent)
        if len(tokens) == 1:
            if tokens[0] in self.word_dict:
                return {"1": tokens[0]}
        else:
            for x in tokens:
                if len(x) > 1 and x in self.word_dict:
                    return {"1": x}

        return {"0": ""}
