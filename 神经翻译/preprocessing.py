# -*- coding: utf-8 -*
"""
Author: Junkai-Yan
File: preprocessing.py
Finished in 2019/12/16
This file pre-process source data set.
Firstly, read in training data set, numbering the word in training set for source language and target language in a dict.
Secondly, convert the sentence to num sequence according to the dict generated before.
Thirdly, read in testing and developing data set, convert the sentence to num sequence according to the dict.
Lastly, write relevant information in txt file.
**********
What you should focus on is that I padding '<BOS>' and '<EOS>' at the beginning and ending of the sentence respectively.
If the word in testing or developing data set is a new word, I replacing it by '<UKN>'.
What's more, there is a word '<PAD>' in my dict, means to padding if the sentences have different length in a batch.
Word segmentation tool is jieba.
"""

import jieba

def open_file_train(filename, word2num):
    """
    Open training file and generate dict, encoding source file's
    sentence to sequence
    :param filename: filename
    :param word2num: dict, key is word, value is number
    :return: None
    """
    f_r = open('./dataset_10000/'+filename, 'r',encoding='utf-8')
    f_w_data = open('./predata_10000/'+filename, 'w',encoding='utf-8')
    f_w_word2num = open('./predata_10000/word2num_'+filename, 'w',encoding='utf-8')
    f_w_num2word = open('./predata_10000/num2word_'+filename, 'w',encoding='utf-8')
    for line in f_r:
        line = line.strip()
        seg_list = jieba.cut(line)
        sentence = '<BOS> ' + " ".join(seg_list) +' <EOS>'
        sentence = sentence.split()
        # generate dict and convert sentence to sequence
        for word in sentence:
            if word not in word2num:
                word2num[word] = len(word2num)
            f_w_data.write(str(word2num[word])+' ')
        f_w_data.write('\n')
    for it in word2num:
        f_w_word2num.write(it+' '+str(word2num[it])+'\n')
        f_w_num2word.write(str(word2num[it])+' '+it+'\n')
    f_r.close()
    f_w_data.close()
    f_w_word2num.close()
    f_w_num2word.close()

def open_file_test(filename, word2num):
    """
    Open testing file, encoding source file's sentence to sequence
    :param filename: filename
    :param word2num: dict, key is word, value is number
    :return: None
    """
    f_r = open('./dataset_10000/'+filename, 'r',encoding='utf-8')
    f_w_data = open('./predata_10000/'+filename, 'w',encoding='utf-8')
    for line in f_r:
        line = line.strip()
        seg_list = jieba.cut(line)
        sentence = '<BOS> ' + " ".join(seg_list) +' <EOS>'
        sentence = sentence.split()
        # convert sentence to sequence
        for word in sentence:
            # if is new word, replacing it by '<UKN>'
            if word not in word2num:
                f_w_data.write(str(word2num['<UKN>']) + ' ')
            else:
                f_w_data.write(str(word2num[word])+' ')
        f_w_data.write('\n')
    f_r.close()
    f_w_data.close()


if __name__=="__main__":
    chinese_word2num = {'<BOS>':0, '<EOS>':1, '<UKN>':2, '<PAD>':3}
    english_word2num = {'<BOS>':0, '<EOS>':1, '<UKN>':2, '<PAD>':3}
    open_file_train('train_source_8000.txt', chinese_word2num)
    open_file_train('train_target_8000.txt', english_word2num)
    open_file_test('test_source_1000.txt', chinese_word2num)
    open_file_test('test_target_1000.txt', english_word2num)
    open_file_test('dev_source_1000.txt', chinese_word2num)
    open_file_test('dev_target_1000.txt', english_word2num)
