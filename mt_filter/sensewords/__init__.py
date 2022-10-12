#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @time: 2022/10/11 
# @Author: czx

# 初始化中英文过滤

import os
from .filters import SenseFilter

current_path = os.path.dirname(os.path.realpath(__file__))

ChineseFilter = SenseFilter(os.path.join(current_path, "dicts/chinese_dict.txt"),
                            os.path.join(current_path, "dicts/chinese_phrase.txt"))
EnglishFilter = SenseFilter(os.path.join(current_path, "./dicts/english_dict.txt"),
                            os.path.join(current_path, "./dicts/english_phrase.txt"), english=True)