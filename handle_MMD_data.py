#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/10/10 22:02
# @Author  : chen
# @FileName: handle_MMD_data
# @Software: PyCharm
import os
import sys

l = os.listdir("MMD_data")
for i in l:
    with open("MMD_data/"+i,encoding='utf-8') as file:
        data = file.readlines()
    for index,j in enumerate(data):
        try:
            sentence,label = j.split('####')
        except:
            print(j)
            sys.exit(3)
        new_label = []
        for a in label.split():
            if a == 'O' or 'OP' in a:
                new_label.append('O')
            else:
                second = a.split('-')[1]
                new_label.append('T'+'-'+second)
        data[index] = sentence + '***' + ' '.join(new_label) + '\n'
    with open("MMD_data/"+'new_'+i,'w',encoding='utf-8') as file:
        file.writelines(data)
