# -*- cooding: utf-8 -*-
import re
"""Yield successive n-sized chunks from l."""
def chunks(l, n, truncate=False):
    batches = []
    for i in range(0, len(l), n):
        if truncate and len(l[i:i + n]) < n:
            continue
        batches.append(l[i:i + n])
    return batches

'''
对文本进行切字处理
'''
regEx = re.compile("[\\W]*")
def get_word_list(s1): # 把句子按字分开，中文按字分，英文按单词，数字按空格
    res = re.compile(r"([\u4e00-\u9fa5])") # [\u4e00-\u9fa5]中文范围
    p1 = regEx.split(s1.lower())
    str1_list = []
    for str in p1:
        if res.split(str) == None:
            str1_list.append(str)
        else:
            ret = res.split(str)
            for ch in ret:
                str1_list.append(ch)
                list_word = [w for w in str1_list if len(w.strip()) > 0] # 去掉为空的字符
    return list_word

'''
根据最大长度对数据进行padding
'''

def padding(l, max_len):
    n = len(l)
    for i in range(max_len - n):
        l.insert(0, 0)
    return l