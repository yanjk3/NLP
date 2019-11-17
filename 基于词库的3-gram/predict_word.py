import re
import jieba
import random

def predict(gram, lib):
    pre = -1
    frequency = 0
    for gram3 in lib:
        if gram[0]==gram3[0] and gram[1]==gram3[1]:
            if lib[gram3] > frequency:
                frequency = lib[gram3]
                pre = gram3[2]
    return pre

word_to_idx = dict()
f = open('./分词结果和词表/dict.txt', 'r', encoding='utf-8')
for line in f:
    l = line.split()
    word_to_idx[l[1]] = int(l[0])
f.close()
word_to_idx['<BOS>'] = len(word_to_idx)
word_to_idx['<EOS>'] = len(word_to_idx)
idx_to_word = {word_to_idx[word]: word for word in word_to_idx}

stop_list = list()
f = open('停止词.txt', 'r')
for line in f:
    stop_list.append(line.strip())
f.close()


gram_lib = dict()
f = open('gram_lib.txt', 'r', encoding='utf-8')
for line in f:
    l = line.split()
    gram_lib[(int(l[0]), int(l[1]), int(l[2]))] = int(l[3])
f.close()

f = open('questions.txt', 'r', encoding='utf-8')
fw = open('3-gram-classic-answer.txt', 'w', encoding='utf-8')
for line in f:
    l = re.split('[、|。|？|！|；|：|，|（|）|~|——|“|”|…|\n]', line)
    sentence = ['<BOS>']
    for s in l:
        if len(s) == 0:
            continue
        s = jieba.cut(s)
        sentence += list(s)
    sentence += ['<EOS>']
    test = list()
    for s in sentence:
        if s not in stop_list:
            if s != '<BOS>' and s != '<EOS>':
                s = re.sub('\W', '', s)
            if s != '':
                test.append(s)
    position = test.index('MASK')
    i, j = test[position-2], test[position-1]
    if i in word_to_idx and j in word_to_idx:
        pre = predict((word_to_idx[i], word_to_idx[j]), gram_lib)
        if pre == -1:
            fw.write(str(idx_to_word[random.randint(0, len(idx_to_word)-1)])+'\n')
        else:
            fw.write(str(idx_to_word[pre])+'\n')
    else:
        fw.write('Error\n')
f.close()
fw.close()